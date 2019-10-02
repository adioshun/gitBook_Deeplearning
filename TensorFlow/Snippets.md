# TF1.x Snippets





---

# TF2.x Snippets


## 1. DataLoad 
- https://github.com/ageron/tf2_course/blob/master/03_loading_and_preprocessing_data.ipynb

```python 
import numpy as np
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
dataset

for item in dataset:
    print(item)

for item in dataset:
    print(item.numpy(), end=" ")

#-----------------
dataset = tf.data.Dataset.from_tensor_slices({"features": X, "label": y})
dataset


for item in dataset:
    print(item["features"].numpy(), item["label"].numpy())
    
```

## 2. model save & Load
- https://github.com/ageron/tf2_course/blob/master/04_deploy_and_distribute_tf2.ipynb


### 2.1 Keras 

```python 
#Save
model_path = './models/keras/model.json'
weights_path = './models/keras/weights.h5'

model.save(model_path)
model.save_weights(weights_path)

#Load
model = keras.models.load_model("my_mnist_model.h5")
model.load_weights("./yolov3") 
# https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/master/4-Object_Detection/YOLOV3/test.py
```

### 2.2 TF

```pythbon 
# Save 
tf.saved_model.save(model, model_path)




# load

loaded_model = tf.saved_model.load(model_path)



```

### 2.3 Numpy

```python 
weighs = np.load("./vgg16.npy", encoding='latin1').item()
for layer_name in weighs.keys():
    layer = model.get_layer(layer_name)
    layer.set_weights(weighs[layer_name])

```


## 3. 평가 

```python 
def evaluate(model, images, labels):
    logits = model(images, training=False)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
    
# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_loss = 0.
    avg_train_acc = 0.
    avg_test_acc = 0.
    train_step = 0
    test_step = 0
    
    for images, labels in train_dataset:
        grads = grad(model, images, labels)                
        optimizer.apply_gradients(zip(grads, model.variables))
        loss = loss_fn(model, images, labels)
        acc = evaluate(model, images, labels)
        avg_loss = avg_loss + loss
        avg_train_acc = avg_train_acc + acc
        train_step += 1
    avg_loss = avg_loss / train_step
    avg_train_acc = avg_train_acc / train_step
    
    for images, labels in test_dataset:        
        acc = evaluate(model, images, labels)        
        avg_test_acc = avg_test_acc + acc
        test_step += 1    
    avg_test_acc = avg_test_acc / test_step    

    print('Epoch:', '{}'.format(epoch + 1), 'loss =', '{:.8f}'.format(avg_loss), 
          'train accuracy = ', '{:.4f}'.format(avg_train_acc), 
          'test accuracy = ', '{:.4f}'.format(avg_test_acc))
    
    checkpoint.save(file_prefix=checkpoint_prefix)

print('Learning Finished!')


```

---


## 1. 모델 zoo

- [Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)

## 2. 저장 하기 

### 2,1 모델 체크포인트 .ckpt 
- 재학습 가능 모델에대한 메타정보 포함 
- 파일 크기가 크다 
- graph.pbtxt : 노드 정보가 모두 기록, .ckpt와 같이 생성 됨, input_graph 옵션의 입력값으로 활용됨 
- `tf.train.Saver().save(sess, 'trained.ckpt')` :학습한 변수 값들을  ckpt 체크포인트로 저장


### 2.2 pb 파일
- 재학습 불가능 
- 메타 데이타는 제외하고 모델과 가중치 값 포함 (모델의 그래프 + 학습된 변수값)
- tensorflow API를 이용한 C++ 프로그램에서 사용하는 포맷
- `tf.train.write_graph(sess.graph_def, ".", 'trained.pb', as_text=False)`: 그래프 저장


###### [참고] trained.ckpt+ trained.pb -> frozen_graph.pb 변환 툴 : [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)
- out_node_names 옵션 : 자기가 사용하고자 하는 모델의 출력 노드 지정, `graph.pbtxt`파일 참고 
- 오류시 파일 상단 설정 부분에 `--input_binary=true` 추가 


> 참고 : [The tensorgraph is a example show how to generate, load graph from tensorflow](https://github.com/JackyTung/tensorgraph)



# Loading and Preprocessing Data
