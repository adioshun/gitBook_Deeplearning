# [Create Network ](https://nbviewer.jupyter.org/github/deeplearningzerotoall/TensorFlow/blob/master/lab-10-1-1-mnist_nn_softmax.ipynb)


[구현방법](https://www.youtube.com/watch?v=OR_NwgouflE&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=36)
1. Keras Sequential API : 블록 쌓듯이 쉽게, 멀티 인붓/아웃풋 불가, shared Layer 불가 
2. KERAS Functional : Input layer 지정으로 멀티 인북 가능 
3. Keras.Model subclassing : Fully-customizable model 생성 가능, `_init_`method에 레이어 선언, `call`method에서 사용 

> 전체 흐름 강의 : [Logistic Regression-강의](https://www.youtube.com/watch?v=enyQpA-xAYc&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=11), [Logistic Regression-코드](https://github.com/deeplearningzerotoall/TensorFlow/blob/master/lab-05-1-logistic_regression-eager.ipynb)

> [TensorFlow 2.0 Examples](https://github.com/aymericdamien/TensorFlow-Examples/tree/master/tensorflow_v2)



> 추천 구조 : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb
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
  
# Store layers weight & bias

# A random value generator to initialize weights.
#random_normal = tf.initializers.RandomNormal()

weights = {
    # Conv Layer 1: 5x5 conv, 1 input, 32 filters (MNIST has 1 color channel only).
    'wc1': tf.Variable(tf.zeros([5, 5, 1, conv1_filters])),
    # Conv Layer 2: 5x5 conv, 32 inputs, 64 filters.
    'wc2': tf.Variable(tf.zeros([5, 5, conv1_filters, conv2_filters])),
    # FC Layer 1: 7*7*64 inputs, 1024 units.
    'wd1': tf.Variable(tf.zeros([7*7*64, fc1_units])),
    # FC Out Layer: 1024 inputs, 10 units (total number of classes)
    'out': tf.Variable(tf.zeros([fc1_units, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.zeros([conv1_filters])),
    'bc2': tf.Variable(tf.zeros([conv2_filters])),
    'bd1': tf.Variable(tf.zeros([fc1_units])),
    'out': tf.Variable(tf.zeros([num_classes]))
}
  
# Create model
def conv_net(x):
    
    # Input shape: [-1, 28, 28, 1]. A batch of 28x28x1 (grayscale) images.
    x = tf.reshape(x, [-1, 28, 28, 1])

    # Convolution Layer. Output shape: [-1, 28, 28, 32].
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    # Max Pooling (down-sampling). Output shape: [-1, 14, 14, 32].
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer. Output shape: [-1, 14, 14, 64].
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    
    # Max Pooling (down-sampling). Output shape: [-1, 7, 7, 64].
    conv2 = maxpool2d(conv2, k=2)

    # Reshape conv2 output to fit fully connected layer input, Output shape: [-1, 7*7*64].
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    
    # Fully connected layer, Output shape: [-1, 1024].
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # Apply ReLU to fc1 output for non-linearity.
    fc1 = tf.nn.relu(fc1)

    # Fully connected layer, Output shape: [-1, 10].
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(out)
  
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



```







---
## 모두 함수화 

```python 
# https://github.com/dragen1860/TensorFlow-2.x-Tutorials/blob/master/05-FashionMNIST/mnist_matmul.py

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

    def call(self, x):


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


def compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

def main():

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






---
## Training by hand



```python 
optimizer = tf.optimizers.Adam()
loss_fn = tf.losses.MeanSquaredError()

@tf.function

def train_step(feature, target):

    with tf.GradientTape() as tape:
        y_pred = model(feature, training=True)
        loss = loss_fn(target, y_pred)
    
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))
    
    return loss

@tf.function

def val_step(feature, target):
    
    y_pred = model(feature)
    loss = loss_fn(target, y_pred)
    
    return loss


for i in range(10):
    
    running_loss = 0
    running_val_loss = 0
    
    for i, (batch_feature, batch_target) in enumerate(dataset):
        loss_ = train_step(batch_feature, batch_target)
        running_loss += loss_
        
    for j, (batch_feature, batch_target) in enumerate(val_dataset):
        loss_ = val_step(batch_feature, batch_target)
        running_val_loss += loss_
        
    print("----------epoch {}--------".format(i+1))
    print("loss: {},  val_loss: {}".format(running_loss/(i+1), 
                                           running_val_loss/(j+1)))


#----------------------------------------------------------------------------


# Keras 에서 accuracy 구하기 
test_loss, test_acc = model.evaluate(x=x_test_eager.numpy(), 
                                     y=y_test_eager.numpy())


# TF에서 Accuracy 구하기 

def accuracy(y, y_pre):
    return tf.keras.metrics.categorical_accuracy(y, y_pre)

for j in range(num_epochs):
    
    running_loss = 0
    running_acc = 0

    for i, (x_, y_) in enumerate(train_dataset):
        
        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                y_pre = model(x_, training=True)
                loss = loss_fn(y_, y_pre)
            acc = accuracy(y_, y_pre)



















