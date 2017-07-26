> Deep residual learning for image recognition, https://arxiv.org/abs/1512.03385

ILSVRC 2015년 대회에서 우승을 한 구조로 마이크로소프트의 Kaiming He 등에 의해서 개발이 되었다.

기존 DNN(Deep Neural Network)보다 layer 수가 훨씬 많은 Deeper NN에 대한 학습(training)을

쉽게 할 수 있도록 해주는 residual framework 개념을 도입했다.


![](http://i.imgur.com/nbOpQGv.png)

## 1. 개요 

### 1.1 깊은 망의 문제점 

-  Vanishing/Exploding Gradient 문제 : batch normalization, 파라미터의 초기값 설정 방법을 쓰지만 여전히 일정 수 이상 깊어 지면 문제점 

-  파라미터 수 증가 : 오버피팅, 

- 에러 증가 (층이 깊어질 수록 모 error가 누적된다)


###### [참고] 깊이에 따른 기술 
- 5층 이상 : ReLU 함수 도입
- 10층 이상 : 학습 파라미터들의 초기화 전략, 배치 정규화
- 30 층 이상과 100 층 이상 : 몇 개의 층을 건너뛰는 연결을 만드는 방법(잔차넷)



### 1.2 Residual Learning
> 참고 : [홍정모 교수 블로그](http://blog.naver.com/atelierjpro/220966166731), [Donghyun](http://blog.naver.com/kangdonghyun/220992404778), [SNU강의자료_29P](https://bi.snu.ac.kr/Courses/ML2016/LectureNote/LectureNote_ch9.pdf)

100 layer 이상으로 깊게 하면서, 깊이에 따른 학습 효과를 얻을 수 있는 방법

![](http://i.imgur.com/Q9kYDvx.png)

Identity shortcut 연결의 장점 
- 깊은 망도 쉽게 최적화가 가능하다.
- 늘어난 깊이로 인해 정확도를 개선할 수 있다.

## 2. 구조 

![](http://i.imgur.com/7tQQHxk.png?1) 

망을 설계하면서 VGGNet의 설계 철학을 많이 이용

- 가능한 convolutional layer는 3x3 kernel을 가지도록 함 

- Feature-map의 크기가 절반으로 작아지는 경우는 연산량의 균형을 맞추기 위해 필터의 수를 두 배로 늘린다.
    - Feature-map의 크기를 줄일 때는 pooling 대신에  stride의 크기를 “2”로


- 복잡도(연산량)를 줄이기 위해 max-pooling(1곳 제외), hidden fc, dropout 등을 사용하지 않았다

- 매 2개의 convolutional layer마다 shortcut connection이 연결되도록 하였다.

![](http://i.imgur.com/undefined.png)

### 2.1 Deeper Bottleneck Architecture

학습 속도 향상을 위해 (50- /101-/152-layer)
- Bottleneck적용 

- residual function은 1x1, 3x3, 1x1

![](http://i.imgur.com/Qqcpcie.png)

## 3. 특징 

## 4. 학습/테스트 

