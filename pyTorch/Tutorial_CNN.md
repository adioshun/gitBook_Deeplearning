


# Tutorial - CNN (MNIST기준)

## 1. 개요 

pyTorch를 이용하여 MNIST 구현 하기 : [Youtube](https://www.youtube.com/watch?v=wQtkdq3tmJ8&list=PLQ28Nx3M4JrhkqBVIXg-i5_CVVoS1UzAv&index=20), [Slide](https://drive.google.com/drive/folders/1qVcF8-tx9LexdDT-IY6qOnHc8ekDoL03)



## 2 데이터 생성 

`DataLoader()` 사용   

- 대용량 데이터 처리를 위해서는 mini-batch로 나누어서 진행 하길 권장 (데이터를 일부분으로 쪼개어서 학습)  
- pyTorch에서는 `DataLoader()`를 이용하여 미니패치 처리 가능 `train_loader = torch.utils.data.DataLoader()`  
- 단, 사용자는 지정된 형식의 반환 method를 구현해야함 : `_len_()`와 `_getitem_()`  
  

![](https://i.imgur.com/IZ8TWUS.png)  
  

```python   
 DataLoader(datatet, # 사용자 만든 class _len_()`와 `_getitem_()` 포함 해야함   
            batch_size = 2, # 각 미니패치 크기, 보통 2의 제곱승 설정   
            shuffle=Ture, #Epoch마다 데이터셋 순서를 썪어서, 학습 순서를 바꾼다.
            drop_last = True)  
```  
  
## 3. Modeling

![](https://i.imgur.com/9tvCcd7.png)

```python 
input = torch.Tensor(1,1,28,28) #batch_size, channel, height, width = N, C, H, W
conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) #입력 채널, 출력 채널, 커널 사이즈, stride, padding
pool = torch.nn.MaxPool2d(kernel_size=2, stride=2))
conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
pool = torch.nn.MaxPool2d(kernel_size=2, stride=2))
fc = fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True) #Final FC 7x7x64 inputs -> 10 outputs
```

```python 
# CNN Model (2 conv layers)
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # Final FC 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc(out)
        return out

# instantiate CNN model
model = CNN()
```

## 4. Loss(=Cost) 정의 하기 

```python 
criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
```

## 5. 학습 (Gradient Descent)


```python 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

```python 

# train my model
total_batch = len(data_loader)
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)  # 이미지 
        Y = Y.to(device)  # 라벨 

        optimizer.zero_grad()  #필수 
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

print('Learning Finished!')
```

## 6. Test 

```python 

# Test model and check accuracy
with torch.no_grad(): #테스트 용이니 학습을 안함 명시 
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
``` 


---

## Layer 추가 해보기 

![](https://i.imgur.com/x95SbhF.png)

```python 

# CNN Model
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        return out
  ```
