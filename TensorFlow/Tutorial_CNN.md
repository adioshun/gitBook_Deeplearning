# [Create Network ](https://nbviewer.jupyter.org/github/deeplearningzerotoall/TensorFlow/blob/master/lab-10-1-1-mnist_nn_softmax.ipynb)


> 전체 흐름 강의 : [Logistic Regression-강의](https://www.youtube.com/watch?v=enyQpA-xAYc&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=11), [Logistic Regression-코드](https://github.com/deeplearningzerotoall/TensorFlow/blob/master/lab-05-1-logistic_regression-eager.ipynb), [mnist cnn keras sequential eager-강의4개](https://www.youtube.com/watch?v=OR_NwgouflE&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=36)


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
    for images, labels in train_dataset:
        grads = grad(model, images, labels)                
        optimizer.apply_gradients(zip(grads, model.variables))
        loss = loss_fn(model, images, labels)
        acc = evaluate(model, images, labels)
        ``` 

