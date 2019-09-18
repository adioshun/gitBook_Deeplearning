# [Create Network ](https://nbviewer.jupyter.org/github/deeplearningzerotoall/TensorFlow/blob/master/lab-10-1-1-mnist_nn_softmax.ipynb)


[구현방법]((https://www.youtube.com/watch?v=OR_NwgouflE&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=36)
1. Keras Sequential API : 블록 쌓듯이 쉽게, 멀티 인붓/아웃풋 불가, shared Layer 불가 
2. KERAS Functional : Input layer 지정으로 멀티 인북 가능 
3. Keras.Model subclassing : Fully-customizable model 생성 가능, `_init_`method에 레이어 선언, `call`method에서 사용 

> 전체 흐름 강의 : [Logistic Regression-강의](https://www.youtube.com/watch?v=enyQpA-xAYc&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=11), [Logistic Regression-코드](https://github.com/deeplearningzerotoall/TensorFlow/blob/master/lab-05-1-logistic_regression-eager.ipynb)

---

## 1. 데이터 준비 


---


## 2. 모델 생성 

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
---

## 3. 실행 (Eager 모드)




```python 
# https://www.tensorflow.org/guide/eager#eager_training

def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])

model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)


# Training loop
grads = grad(model, training_inputs, training_outputs)
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






























