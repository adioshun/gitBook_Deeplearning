# Tensorflow - Linear Regression 

> [youtube](https://youtu.be/TvNd1vNEARw),  [Jupyter](https://github.com/deeplearningzerotoall/TensorFlow/blob/master/lab-02-1-Simple-Linear-Regression-eager.ipynb)

## 1. 개요 

 공부 시간(1,2,3시간)과 점수 (2,4,6점) 를 학습 하여 예상 공부시간 (4시간)일때의 점수를 예측 한다. 

## 2. 데이터 생성 

```python  
## Linear Regression에서는
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5] #Continuous 


## Logistic Regression의 경우
x_train = [[1., 2.],
          [2., 3.],
          [3., 1.],
          [4., 3.],
          [5., 3.],
          [6., 2.]]
y_train = [[0.], 
          [0.],
          [0.],
          [1.],
          [1.],
          [1.]] #Discrete

## multi-class classification(Softmax)의 경우 
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]] #One hot 인코딩 
          
```

## 3. Modeling

```python 
# Weight Initialization (2.0, 0.5)
W = tf.Variable(2.0)
b = tf.Variable(0.5)

hypothesis = W * x_data + b

"""
# Multivariate Linear Regression에서는 
hypothesis = W1 * x1_data + W2 * x2_data + b #기본
hypothesis = tf.matmul(W, x_data) + b # Matrix 연산, W와 x_data 위치 변화 조심 

# Logistic Regression의 경우 tf.exp 시그모이드 함수 이용 
hypothesis  = tf.div(1., 1. + tf.exp(tf.matmul(features, W) + b))

# Multi-class classification 나온값을 SoftMax합수에 넣어 확률값으로 출력 
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b) # softmax = exp(logits) / reduce_sum(exp(logits), dim)
"""
```


## 4. Loss(=Cost) 정의 하기 

```python 
# Linear Regression의 경우 MSE(Mean Squared Error)로 Loss계산 
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Logistic Regression의 경우 
cost = -tf.reduce_mean(labels * tf.log(logistic_regression(features)) + (1 - labels) * tf.log(1 - hypothesis))

# multi-class classification에서는 
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(logits), axis=1)) 

logits = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
```


![](https://i.imgur.com/gvGEc2J.png)


## 5. 학습 (Gradient Descent)


![](https://i.imgur.com/YwEuMza.png)
cost()를 미분하여 기울기를 구하는 문제 



```python 
with tf.GradientTape() as tape:  #변수들의 변화를 tape에 저장 
    hypothesis = W * x_data + b
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

W_grad, b_grad = tape.gradient(cost, [W, b])  #미분, 기울기값을 반환 

#파라미터 업데이트 
learning_rate = 0.01

W.assign_sub(learning_rate * W_grad) # A.assign_sub(B) : A = A-B : A-+ B
b.assign_sub(learning_rate * b_grad)
```


```python 
def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.log(logistic_regression(features)) + (1 - labels) * tf.log(1 - hypothesis))
    return cost
    
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features),features,labels)
    return tape.gradient(loss_value, [W,b])
```


## 6. 전체 코드 

### 6.1 Linear Regression의 경우
```python 
W = tf.Variable(2.9)
b = tf.Variable(0.5)

for i in range(100):
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
      print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))
  ```

### 6.2 Logistic Regression의 경우 

```python 

EPOCHS = 1001

for step in range(EPOCHS):
    for features, labels  in tfe.Iterator(dataset):
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features),features,labels)))
```


## 7. 테스트 

```python 

def prediction(X, Y):
    pred = tf.argmax(hypothesis(X), 1)
    correct_prediction = tf.equal(pred, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
```
