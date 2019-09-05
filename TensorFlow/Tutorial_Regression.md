# Tensorflow - Linear Regression 

> [youtube](https://youtu.be/TvNd1vNEARw),  [Jupyter](https://github.com/deeplearningzerotoall/TensorFlow/blob/master/lab-02-1-Simple-Linear-Regression-eager.ipynb)

## 1. 개요 

 공부 시간(1,2,3시간)과 점수 (2,4,6점) 를 학습 하여 예상 공부시간 (4시간)일때의 점수를 예측 한다. 

## 2 데이터 생성 

```python  
x_train  =  torch.FloatTensor([[1],  [2],  [3]])  
y_train  =  torch.FloatTensor([[2],  [4],  [5]])
```

## 3. Modeling

```python 
# Weight Initialization
W = torch.zeros(1, requires_grad=True) #requires_grad = 학습용 값임을 명시 
# W = torch.zeros((4,3), requires_grad=True) #input(Feature?) dim =4, Class =3 (4개의 정보를 입력 받아 3개의 값으로 매칭)
b = torch.zeros(1, requires_grad=True)

hypothesis  =  x_train  *  W  +  b

"""
# Multivariate Linear Regression에서는 
hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b #기본
hypothesis = x_train.matmul(W) + b # 간단

# Logistic Regression에서는 Regression으로 나온값을 sigmoid합수에 넣어 0 or 1을 출력 
hypothesis = torch.sigmoid(x_train.matmul(W) + b)

# Multi-class classification 나온값을 SoftMax합수에 넣어 확률값으로 출력 
hypothesis = F.softmax(z, dim=0)
"""
```


## 4. Loss(=Cost) 정의 하기 

```python 
# Linear Regression의 경우 MSE(Mean Squared Error)로 Loss계산 
cost = torch.mean((hypothesis - y_train) ** 2)

# Logistic Regression의 경우 Binary Cross Entropy로 계산 
cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()

# multi-class classification에서는 
y_one_hot  =  torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
cost  =  (y_one_hot  *  -torch.log(hypothesis)).sum(dim=1).mean()

```


![](https://i.imgur.com/gvGEc2J.png)


## 5. 학습 (Gradient Descent)


![](https://i.imgur.com/YwEuMza.png)
cost()를 미분하여 기울기를 구하는 문제 

```python 
optimizer = optim.SGD([W, b], lr=0.01)

#항상 붙어 다니는 3줄 
optimizer.zero_grad()  # gradient 초기화 
cost.backward()        # gradient 계산  
optimizer.step()       # gradient 개선(=descent)
```

## 6. 전체 코드 

```python 
 # 데이터
 x_train = torch.FloatTensor([[1], [2], [3]])
 y_train = torch.FloatTensor([[1], [2], [3]])
 # 모델 초기화
 W = torch.zeros(1, requires_grad=True)
 b = torch.zeros(1, requires_grad=True)
 # optimizer 설정
 optimizer = optim.SGD([W, b], lr=0.01)
    
nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train * W + b
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()  #경사 초기화 
    cost.backward() #역전파 계산
    optimizer.step() #가중치 업데이트 

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
  ```

 ---
# nn.Module 을 이용한 코드 작성 

## 1. 개요 

> 기본적으로 PyTorch의 모든 모델은 제공되는 `nn.Module`을 inherit 해서 만들게 됩니다.


## 2 데이터 생성 

> 동일 

```python  
x_train  =  torch.FloatTensor([[1],  [2],  [3]])  
y_train  =  torch.FloatTensor([[2],  [4],  [5]])
```

## 3. Modeling

```python 
class LinearRegressionModel(nn.Module):
    def __init__(self):  #사용할 레이어(nn.Linear)들을 정의
        super().__init__()
        self.linear = nn.Linear(1, 1)  # (입력차원, 출력차원)

    def forward(self, x):  #`forward`에서는 이 모델이 어떻게 입력값에서 출력값을 계산하는지 알려줍니다.
        return self.linear(x) #hypothesis 정의 
        #return self.sigmoid(self.linear(x)) # Logistic Regression시 

model  =  LinearRegressionModel()
 
hypothesis  =  model(x_train)
``` 

## 4. Loss(=Cost) 정의 하기 

```python 
# pytorch 에서 제공 (torch.nn.functional)
cost = F.mse_loss(hypothesis,  y_train)

# Logistic Regression에서는 
cost = F.binary_cross_entropy(hypothesis, y_train)

# multi-class classification에서는 
cost = F.nll_loss(F.log_softmax(z, dim=1), y) # nll = Negative Log Likelihood 
## or 
cost = F.cross_entropy(z,  y)  #Combines F.log_softmax()` and `F.nll_loss()`.
```

## 5. 학습 (Gradient Descent)

> 동일 

```python 
optimizer = optim.SGD([W, b], lr=0.01)

#항상 붙어 다니는 3줄 
optimizer.zero_grad()  # gradient 초기화 
cost.backward()        # gradient 계산  
optimizer.step()       # gradient 개선(=descent)
```

## 6. 전체 코드 


```python 
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
model = LinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        params = list(model.parameters())
        W = params[0].item()
        b = params[1].item()
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W, b, cost.item()
        ))
  ```
