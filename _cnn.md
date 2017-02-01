# CNN 

## 1. 기본 개념 
* 입력을 나누어서 받음- 이미지의 일부부만(filter) 받음

```
기본 아이디어 
* 고양이가 이미지를 보때 뇌파 관측 
* 이미지에 따라서 활성화 되는 뉴런들이 다름 확인(=입력을 나누어서 받음)
* 이를 NN에 적용한것이 CNN
```

## 2. 구조 
![](/assets/CNN.PNG)

### 2.1 Conv Layer
1. 7x7(색상) 전체 이미지 준비 
2. 3x3의 filter를 이용해서 이미지의 일부부만 입력 받아 `하나의 값(=Wx+b)`으로 변환
3. 같은 filter(w값이 같음)를 가지고 이미지의 다른 부분도 입력 받음
    * Pad를 이용하여 이미지 작아짐 문제 해결 
4. 다른 filter(w값이 다름)을 이용하여 이미지의 다른 부분도 입력 받음
5. Activation map구성 

### 2.2 RELU Layer

### 2.3 Pooling Layer (=sampling)
![](/assets/maxpooling.PNG)
* 4x4그림에서 2x2필터를 이용하여 2stride만큼 이용하면
* 2x2의 결과 나옴, 이 결과를 어떻게 결정 하느냐가 Pooling(=sampling)
* eg. 위 그림 예시는 `Max Pooling`으로 가장 큰값을 선택 
    * 1,1,5,6 - 6
    * 2,4,7,8 - 8
    * 3,2,1,2 - 3
    * 1,0,3,4 - 4 

### 2.4 FC Layer(Fully Connected) 
* 마지막 Pooling 해서 나온 값을 일반적 레이어를 구성하여 Label을 선택 하도록 함 
https://youtu.be/2-75C-yZaoA?t=4m18s


## 3. 데모 
* CIFAR-10를 이용한 [데모 사이트](http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html) 
* 실시간 학습 결과를 확인 하여 CNN을 이해 하기 편함 

## 4. Case Study 
* [상세 설명 이미지 보기](https://youtu.be/KbNbWTnlYXs?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm)
* 최종적으로 하나씩 파악 하기!!! 
* 새로운 나만의 CNN 만들기 

### 4.1 LeNet-5
* LeCun et al., 1998
* 손글씨 인식에 사용 

### 4.2 AlexNet  
* Krzhevsky et al., 2101
* 이미지 분석에 사용
* First use of ReLU

### 4.3 GoogLeNet
* Szegedy et.al., 2014
* Inception module
* ILSVRC 2014 winner

### 4.4 ResNet
* He et al.,2015
* ILSVRC 2015 winner (3.6% top 5 error)
* Fast Forward 사용

### 4.5 CNN for Sentence Classification 
* Yoon Kim, 2014
* 텍스트 처리 

### 4.6 DeepMind's AlphaGo
* 19x19x48 input (바둑판)
* CONV1 : 5x5 filter, Stride 1, pad 2
* CONV2 : .....
--- 

###### [참고] Pad 
* Filter의 이동하는 크기(=Stride)에 따라서 뽑아 낼수 있는 수의 갯수가 달라짐 
![](/assets/stride.PNG)

* stride의 크기가 증가 할수로 Output이 작아짐(=정보를 잃어버림)
* Pad(테두리를 0으로)을 이용하여 문제 해결 

![](/assets/pad.PNG)