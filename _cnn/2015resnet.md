# ResNet

> Deep residual learning for image recognition, https://arxiv.org/abs/1512.03385

ILSVRC 2015년 대회에서 우승을 한 구조로 마이크로소프트의 Kaiming He 등에 의해서 개발이 되었다.

기존 DNN(Deep Neural Network)보다 layer 수가 훨씬 많은 Deeper NN에 대한 학습(training)을

쉽게 할 수 있도록 해주는 residual framework 개념을 도입했다.

## 1. 개요 

### 1.1 깊은 망의 문제점 

-  Vanishing/Exploding Gradient 문제 : batch normalization, 파라미터의 초기값 설정 방법을 쓰지만 여전히 일정 수 이상 깊어 지면 문제점 

-  파라미터 수 증가 : 오버피팅, 

- 에러 증가 (층이 깊어질 수록 training error가 누적된다)

![](http://i.imgur.com/iY7SVIR.png)


###### [참고] 깊이에 따른 기술 
- 5층 이상 : ReLU 함수 도입
- 10층 이상 : 학습 파라미터들의 초기화 전략, 배치 정규화
- 30 층 이상과 100 층 이상 : 몇 개의 층을 건너뛰는 연결을 만드는 방법(잔차넷)



### 1.2 Residual Learning
> 참고 : [홍정모 교수 블로그](http://blog.naver.com/atelierjpro/220966166731), [Donghyun](http://blog.naver.com/kangdonghyun/220992404778), [SNU강의자료_29P](https://bi.snu.ac.kr/Courses/ML2016/LectureNote/LectureNote_ch9.pdf)

100 layer 이상으로 깊게 하면서, 깊이에 따른 학습 효과를 얻을 수 있는 방법

