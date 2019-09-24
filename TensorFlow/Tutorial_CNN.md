# [Create Network ](https://nbviewer.jupyter.org/github/deeplearningzerotoall/TensorFlow/blob/master/lab-10-1-1-mnist_nn_softmax.ipynb)


[구현방법](https://www.youtube.com/watch?v=OR_NwgouflE&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=36)
1. Keras Sequential API : 블록 쌓듯이 쉽게, 멀티 인붓/아웃풋 불가, shared Layer 불가 
2. KERAS Functional : Input layer 지정으로 멀티 인북 가능 
3. Keras.Model subclassing : Fully-customizable model 생성 가능, `_init_`method에 레이어 선언, `call`method에서 사용 

> 전체 흐름 강의 : [Logistic Regression-강의](https://www.youtube.com/watch?v=enyQpA-xAYc&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=11), [Logistic Regression-코드](https://github.com/deeplearningzerotoall/TensorFlow/blob/master/lab-05-1-logistic_regression-eager.ipynb)

> [TensorFlow 2.0 Examples](https://github.com/aymericdamien/TensorFlow-Examples/tree/master/tensorflow_v2)

---

## 1. 데이터 준비 

```python 
# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
```

---


## 2. 모델 생성 


- TF Low API : tf.nn.conv2
- TF High API : tf.layers.conv2 OR tf.keras.layers.conv2

### 2.1 Class스타일 모델 생성 with Keras

```python 

class create_model_class(tf.keras.Model):
    def __init__(self, label_dim):  #label_dim 분류 클래스 갯수를 유동적으로 받기 위하여 
        super(create_model_class, self).__init__()
        weight_init = tf.keras.initializers.RandomNormal()

        self.model = tf.keras.Sequential()  
        self.model.add(flatten())   # 추후 Fully Connected Layer 사용하기 위하여 펴쳐줌 [N, 28, 28, 1] -> [N, 784]
                                    # CNN을 이용한다면 굳이 필요 없음 
        for i in range(2):
            self.model.add(dense(256, weight_init)) # Fully Connected Layer
            self.model.add(sigmoid())

        self.model.add(dense(label_dim, weight_init))

    def call(self, x, training=None, mask=None):
        x = self.model(x)

        return x
        
        
  ```

### 2.2 function스타일 모델 생성 with Keras   
      
  ```python 
  
  def create_model_function(label_dim) :
    weight_init = tf.keras.initializers.RandomNormal()

    model = tf.keras.Sequential()
    model.add(flatten())

    for i in range(2) :
        model.add(dense(256, weight_init))
        model.add(sigmoid())

    model.add(dense(label_dim, weight_init))

    return model
  
```
  
  
### 2.3 [Model Subclass](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb)

```python 

from tensorflow.keras import Model, layers

# Create TF Model.
class ConvNet(Model):
    # Set layers.
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(1024)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

# Build neural network model.
conv_net = ConvNet()
```
---

## 3. 실행 (Eager 모드)

AllInOne : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb

Build Custom Layers & Modules : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/4_Utils/build_custom_layers.ipynb

```python 
# https://www.tensorflow.org/guide/eager#eager_training

def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])  #tape.gradient(y, x) 명령으로 변수형 텐서 x에 대한 y의 미분값 계산

model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)


# Training loop
grads = grad(model, training_inputs, training_outputs)
#grad = tfe.implicit_gradients(loss_fn) 
## https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network_eager_api.py
optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                            global_step=tf.train.get_or_create_global_step())

```


---


  
```python 
dataset = tf.data.Dataset.from_tensor_slices((data.train.images,
                                              data.train.labels))
...
for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
  ...
  with tfe.GradientTape() as tape:
    logits = model(images, training=True)
    loss_value = loss(logits, labels)
  ...
  grads = tape.gradient(loss_value, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
```












---

## 모두 함수화 

```python 
# https://github.com/dragen1860/TensorFlow-2.x-Tutorials/blob/master/05-FashionMNIST/mnist_matmul.py

import  os
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets

def prepare_mnist_features_and_labels(x, y):
..
  return x, y



def mnist_dataset():
..
  return ds






def compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))


def compute_accuracy(logits, labels):
  predictions = tf.argmax(logits, axis=1)
  return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def train_one_step(model, optimizer, x, y):

  with tf.GradientTape() as tape:

    logits = model(x)
    loss = compute_loss(logits, y)

  # compute gradient
  grads = tape.gradient(loss, model.trainable_variables)
  # update to weights
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  accuracy = compute_accuracy(logits, y)

  # loss and accuracy is scalar tensor
  return loss, accuracy


def train(epoch, model, optimizer):

  train_ds = mnist_dataset()
  loss = 0.0
  accuracy = 0.0

  for step, (x, y) in enumerate(train_ds):

    loss, accuracy = train_one_step(model, optimizer, x, y)

    if step%500==0:
      print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
  return loss, accuracy





class MyLayer(layers.Layer):


    def __init__(self, units):
        """

        :param units: [input_dim, h1_dim,...,hn_dim, output_dim]
        """
        super(MyLayer, self).__init__()


        for i in range(1, len(units)):
            # w: [input_dim, output_dim]
            self.add_variable(name='kernel%d'%i, shape=[units[i-1], units[i]])
            # b: [output_dim]
            self.add_variable(name='bias%d'%i,shape=[units[i]])



    def call(self, x):
        """

        :param x: [b, input_dim]
        :return:
        """
        num = len(self.trainable_variables)

        x = tf.reshape(x, [-1, 28*28])

        for i in range(0, num, 2):

            x = tf.matmul(x, self.trainable_variables[i]) + self.trainable_variables[i+1]

        return x



def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

    train_dataset = mnist_dataset()

    model = MyLayer([28*28, 200, 200, 10])
    for p in model.trainable_variables:
        print(p.name, p.shape)
    optimizer = optimizers.Adam()

    for epoch in range(20):
        loss, accuracy = train(epoch, model, optimizer)

    print('Final epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())


if __name__ == '__main__':
    main()



```


## 메인 함수에 포함 


```python 
import  os
import  tensorflow as tf
from    tensorflow import  keras
from    tensorflow.keras import datasets, layers, optimizers
import  argparse
import  numpy as np
from    network import VGG16




def normalize(X_train, X_test):
..
    return X_train, X_test

def prepare_cifar(x, y):
..
    return x, y



def compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

def main():

    tf.random.set_seed(22)

    print('loading data...')
    (x,y), (x_test, y_test) = datasets.cifar10.load_data()
    x, x_test = normalize(x, x_test)
    print(x.shape, y.shape, x_test.shape, y_test.shape)
    # x = tf.convert_to_tensor(x)
    # y = tf.convert_to_tensor(y)
    train_loader = tf.data.Dataset.from_tensor_slices((x,y))
    train_loader = train_loader.map(prepare_cifar).shuffle(50000).batch(256)

    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_loader = test_loader.map(prepare_cifar).shuffle(10000).batch(256)
    print('done.')




    model = VGG16([32, 32, 3])


    # must specify from_logits=True!
    criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = keras.metrics.CategoricalAccuracy()

    optimizer = optimizers.Adam(learning_rate=0.0001)


    for epoch in range(250):

        for step, (x, y) in enumerate(train_loader):
            # [b, 1] => [b]
            y = tf.squeeze(y, axis=1)
            # [b, 10]
            y = tf.one_hot(y, depth=10)

            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(y, logits)
                # loss2 = compute_loss(logits, tf.argmax(y, axis=1))
                # mse_loss = tf.reduce_sum(tf.square(y-logits))
                # print(y.shape, logits.shape)
                metric.update_state(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            # MUST clip gradient here or it will disconverge!
            grads = [ tf.clip_by_norm(g, 15) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 40 == 0:
                # for g in grads:
                #     print(tf.norm(g).numpy())
                print(epoch, step, 'loss:', float(loss), 'acc:', metric.result().numpy())
                metric.reset_states()


        if epoch % 1 == 0:

            metric = keras.metrics.CategoricalAccuracy()
            for x, y in test_loader:
                # [b, 1] => [b]
                y = tf.squeeze(y, axis=1)
                # [b, 10]
                y = tf.one_hot(y, depth=10)

                logits = model.predict(x)
                # be careful, these functions can accept y as [b] without warnning.
                metric.update_state(y, logits)
            print('test acc:', metric.result().numpy())
            metric.reset_states()







if __name__ == '__main__':
    main()


```


## model - comple - fit -evaluate process

```python 

import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras


# In[1]:


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')




(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
# [b, 28, 28] => [b, 28, 28, 1]
x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
# one hot encode the labels. convert back to numpy as we cannot use a combination of numpy
# and tensors as input to keras
y_train_ohe = tf.one_hot(y_train, depth=10).numpy()
y_test_ohe = tf.one_hot(y_test, depth=10).numpy()

# In[2]:


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)



# 3x3 convolution
def conv3x3(channels, stride=1, kernel=(3, 3)):
    return keras.layers.Conv2D(channels, kernel, strides=stride, padding='same',
                               use_bias=False,
                            kernel_initializer=tf.random_normal_initializer())

class ResnetBlock(keras.Model):

    def __init__(self, channels, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path

        self.conv1 = conv3x3(channels, strides)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = conv3x3(channels)
        self.bn2 = keras.layers.BatchNormalization()

        if residual_path:
            self.down_conv = conv3x3(channels, strides, kernel=(1, 1))
            self.down_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        residual = inputs

        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # this module can be added into self.
        # however, module in for can not be added.
        if self.residual_path:
            residual = self.down_bn(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.down_conv(residual)

        x = x + residual
        return x


class ResNet(keras.Model):

    def __init__(self, block_list, num_classes, initial_filters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.num_blocks = len(block_list)
        self.block_list = block_list

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3x3(self.out_channels)

        self.blocks = keras.models.Sequential(name='dynamic-blocks')

        # build all the blocks
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):

                if block_id != 0 and layer_id == 0:
                    block = ResnetBlock(self.out_channels, strides=2, residual_path=True)
                else:
                    if self.in_channels != self.out_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResnetBlock(self.out_channels, residual_path=residual_path)

                self.in_channels = self.out_channels

                self.blocks.add(block)

            self.out_channels *= 2

        self.final_bn = keras.layers.BatchNormalization()
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)

    def call(self, inputs, training=None):

        out = self.conv_initial(inputs)

        out = self.blocks(out, training=training)

        out = self.final_bn(out, training=training)
        out = tf.nn.relu(out)

        out = self.avg_pool(out)
        out = self.fc(out)


        return out

# In[3]:

def main():
    num_classes = 10
    batch_size = 32
    epochs = 1

    # build model and optimizer
    model = ResNet([2, 2, 2], num_classes)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.build(input_shape=(None, 28, 28, 1))
    print("Number of variables in the model :", len(model.variables))
    model.summary()

    # train
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test_ohe), verbose=1)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)




if __name__ == '__main__':
    main()

```



























