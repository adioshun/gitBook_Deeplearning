# Tensorflow Dataloader


## 1. Numpy 

```python
'''
    ModelNet dataset. Support ModelNet40, ModelNet10, XYZ and normal channels. Up to 10000 points.
'''

class ModelNetDataset():
    def __init__(self, root, batch_size = 32, npoints = 1024, split='train', normalize=True, normal_channel=False, modelnet10=True, cache_size=15000, shuffle=None):
        self.root = '/workspace/datasets/modelnet40_normal_resampled/'
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
            """
            {'bathtub': 0,
            'bed': 1,
            ...
            'toilet': 9}
            """
        self.normal_channel = normal_channel
        
        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        assert(split=='train' or split=='test')
        """
        ['bathtub_0001',
        'bathtub_0002',
        ...
        """
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        """
        ['bathtub',
        'bathtub',
        ...
        """
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]
        """
        ('bathtub',
        '/workspace/datasets/modelnet40_normal_resampled/bathtub/bathtub_0001.txt')
        """
        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (point_set, cls) tuple

        if shuffle is None:
            if split == 'train': self.shuffle = True
            else: self.shuffle = False
        else:
            self.shuffle = shuffle

        self.reset()

    def _augment_batch_data(self, batch_data):   
        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data
        return provider.shuffle_points(rotated_data)


    def _get_item(self, index): 
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1],delimiter=',').astype(np.float32)
            # Take the first npoints
            point_set = point_set[0:self.npoints,:]
            if self.normalize:
                point_set[:,0:3] = pc_normalize(point_set[:,0:3])
            if not self.normal_channel:
                point_set = point_set[:,0:3]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
        return point_set, cls
        
    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.datapath)

    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 3

    def reset(self):
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath)+self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.datapath))
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.npoints, self.num_channel()))
        batch_label = np.zeros((bsize), dtype=np.int32)
        for i in range(bsize):
            ps,cls = self._get_item(self.idxs[i+start_idx])
            batch_data[i] = ps
            batch_label[i] = cls
        self.batch_idx += 1
        if augment: batch_data = self._augment_batch_data(batch_data)
        return batch_data, batch_label
```

```python 
TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, \
                npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, \    
                npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

def train_one_epoch(sess, ops, train_writer):
    ...
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)

    ...
    
    
    TRAIN_DATASET.reset(
```



---

## 2. [tf.data API 이용](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/datasets.md)

### 2.1 tf.data.Dataset 

##### 가. Creating a source 
    - tf.data.Dataset.from_tensors()
    - tf.data.dataset.from_tensor_slices(np_features, np_labels)

###### - 바로 입력 (2GB제한) 
```python 
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```

###### - placeholder로 입력 

```python 
features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
## 사용시 
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})

```

###### - 파일명 / 전처리 후 입력 
```python 

def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label
  
filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
labels = [0, 37, 29, 1, ...]

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))


```

###### - Batch로 입력 
```python 
batched_dataset = dataset.batch(4)
#`Dimension(None) OR (??) `에러시 https://github.com/tensorflow/tensorflow/issues/18226
# batched_dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(32)) 

# Dim 확인 
print(dataset.output_types) 
print(dataset.output_shapes) 

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()
```

###### 나. Applying a transformation : transform Dataset into a new Dataset
    - (e.g. Dataset.batch())
    - Dataset.map()



### 2.2 tf.data.Iterator 

> The most common way to consume values from a Dataset is to make an iterator object


- Dataset.make_one_shot_iterator()  #`Estimator`에서 유일게 지원 
- Iterator.initializer # feed-dict과 쓰일때 유용 `sess.run(iterator.initializer, feed_dict={max_value: 10})`
- Iterator.get_next()


```python 
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

---

