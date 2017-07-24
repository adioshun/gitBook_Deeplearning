# CNN 

## 1. 개요 

### 1.1 기존 네트워크(MLP)의 문제점 

- MLP는 모든 입력이 위치와 상관없이 동일한 수준의 중요도를 갖는다고 보기 때문

- 이미지가 조금만 변해도(크기, 회전, 변형) 새로운 학습 데이터로 처리를 해줘야 하는 문제점

- 기존의 MLP(multi-layered neural network)이용 이미지 처리시 아래 문제 발생 
    - 학습 시간(Training time)
    - 망의 크기(Network size)
    - 변수의 개수(Number of free parameters)

### 1.2 해결책 

* 입력을 나누어서 받음- 이미지의 일부부만(filter) 받음

```
기본 아이디어 
* 고양이가 이미지를 보때 뇌파 관측 
* 이미지에 따라서 활성화 되는 뉴런들이 다름 확인(=입력을 나누어서 받음)
* 이를 NN에 적용한것이 CNN
```


### 1.3 CNN 특징 

- Locality(Local Connectivity): receptive field와 유사하게 local 정보를 활용

- Shared Weights : 동일한 filter를 전체 영상에 반복적으로 적용
    - 장점 : 변수의 수 감소,  topology 변화에 무관한 항상성(invariance)를 얻을 수 있음 

- 이동이나 변형 등에 무관한(= global한 특징) 학습 결과 : (convolution + sub-sampling) 과정을 여러 번 반복 
    - feature map의 크기가 작아지면서 전체를 대표할 수 있는 강인한 특징들만 남게 된다.

>  수용영역(Receptive Field): 외부 자극이 전체 영향을 끼치는 것이 아니라 특정 영역에만 영향을 준다는 뜻



## 2. 구조 
![](/assets/CNN.PNG)

### 2.1 Conv Layer
![](/assets/onenum.png)
1. 7x7(색상) 전체 이미지 준비 
2. 3x3의 filter를 이용해서 이미지의 일부부만 입력 받아 `하나의 값(=Wx+b)`으로 변환
3. 같은 filter(w값이 같음)를 가지고 이미지의 다른 부분도 입력 받음
    * Pad를 이용하여 이미지 작아짐 문제 해결 
4. 다른 filter(w값이 다름)을 이용하여 이미지의 다른 부분도 입력 받음
5. Activation map구성(각 필터를 통해 생성된 값들)

###### 최종 구하게 되는 값의 크기는?
![](/assets/stride.PNG)
* Filter의 이동하는 크기(=Stride)에 따라서 뽑아 낼수 있는 수의 갯수가 달라짐 
* stride의 크기가 증가 할수록 Output이 작아짐(=정보를 잃어버림)

###### Pad를 통해 최종 구하는 값의 크기를 동일시 하기
* Pad(테두리를 0으로)을 이용하여 문제 해결 
* Activation Maps크기 

![](/assets/pad.PNG)

### 2.2 RELU Layer
* 기존 자료 참고 

### 2.3 Pooling Layer (=sampling)
![](/assets/maxpooling.PNG)
* 4x4그림에서 2x2필터를 이용하여 2 stride만큼 이용하면
* 2x2의 결과 나옴, 이 결과를 어떻게 결정 하느냐가 Pooling(=sampling)
* eg. 위 그림 예시는 `Max Pooling`으로 가장 큰값을 선택 
    * 1,1,5,6 - 6
    * 2,4,7,8 - 8
    * 3,2,1,2 - 3
    * 1,0,3,4 - 4 

### 2.4 FC Layer(Fully Connected) 
* 마지막 Pooling 해서 나온 값을 일반적 레이어를 구성하여 Label을 선택 하도록 함 
https://youtu.be/2-75C-yZaoA?t=4m18s


### 2.5 정규화층
이미지 정규화 방법 
* 통계적 처리
* Local Contrast Normalization(국소 콘트라스트 정규화)

#### A. 국소 콘트라스트 정규화 
###### 감산 정규화 
딥러닝 제대로 시작하기 116page

###### 제산 정규화 
딥러닝 제대로 시작하기 

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
* Krzhevsky et al., 2012
* 이미지 분석에 사용
* First use of ReLU
* ILSVRC 2012

### 4.3 GoogLeNet
* Szegedy et.al., 2014
* Inception module
* ILSVRC 2014 winner

### 4.4 VGG
* ILSVRC 2014
* 19 Layer

### 4.5 ResNet
* He et al.,2015
* ILSVRC 2015 winner (3.6% top 5 error)
* 152 Layer
* Fast Forward 적용
 * VGG보다 레이어가 많지만 더 따른 이유 
 * 8GPU로 2~3주 Training

### 4.6 CNN for Sentence Classification 
* Yoon Kim, 2014
* 텍스트 처리 (일반적으로 RNN을 사용)

### 4.7 DeepMind's AlphaGo
* 19x19x48 input (바둑판)
* CONV1 : 5x5 filter, Stride 1, pad 2
* CONV2 : .....


--- 

