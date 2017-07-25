# AlexNet

> ImageNet Classification with Deep Convolutional Neural Networks, 2012

## 1. 개요

컴퓨터 비전 분야에서 CNN이 널리 적용되도록 한 계기가 된 구조로,

CNN 분야에서 유명한 Krizhevsky와 Hinton 등에 의해 개발이 되었다.

2012년 ImageNet ILSVRC 대회에서

2위와 큰 성능차(AlexnNet의 에러율은 16%이고 2위의 에러율은 26%)를 보이며 우승한 것으로 유명하다.

## 2. 네트워크 구조  

AlexNet의 구조는 LeNet과 유사하지만,

보통 convolutional layer 다음에 pooling(sub-sampling) layer가 오는 기본 구조와 달리,

convolutional layer 바로 뒤에 convolutional layer가 온 점이 특이하다.

![](http://i.imgur.com/LJPDB41.png)

위아래로 나누어져 있는것은 2개의 GPU를 기반으로 한 병렬 구조여서 그렇다. 기본구조는 LeNet5와 크게 다르지 않다.
- GPU-1에서는 주로 컬러와 상관없는 정보를 추출하기 위한 kernel이 학습이 되고,
- GPU-2에서는 주로 color에 관련된 정보를 추출하기 위한 kernel이 학습이 된다. 

## 3. 특징

### 3.1 ReLU

### 3.2 overlapped pooling

### 3.3 response normalization

### 3.4 dropout 

### 3.5 2개의 GPU 사용이라고 볼 수 있다.