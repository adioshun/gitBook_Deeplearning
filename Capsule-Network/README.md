# Capsule Networks

![](https://i.imgur.com/E7dohf2.png)

## 1. List



## 2. Paper

- [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829): arXiv 2017, 10,26

- [MATRIX CAPSULES WITH EM ROUTING](https://openreview.net/pdf?id=HJWLfGWRb): Under review as a conference paper at ICLR 2018

## 3. Article (Post, blog, etc.)

- [GOOGLE’S AI WIZARD UNVEILS A NEW TWIST ON NEURAL NETWORKS](https://www.wired.com/story/googles-ai-wizard-unveils-a-new-twist-on-neural-networks/)

- ["What is wrong with convolutional neural nets ?"](http://moreisdifferent.com/2017/09/hinton-whats-wrong-with-CNNs):Geoffrey Hinton talk,  

- [What is a CapsNet or Capsule Network?](https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc)

- [Dynamic Routing Between Capsules - 캡슐 간 동적 라우팅](http://blog.naver.com/sogangori/221129974140): 딥러닝 강

- [A Visual Representation of Capsule Network Computations](https://medium.com/@mike_ross/a-visual-representation-of-capsule-network-computations-83767d79e737)

## 3. Tutorial (Series, )



## 4. Youtube

- ["What is wrong with convolutional neural nets ?"](https://www.youtube.com/watch?v=rTawFwUvnLE&feature=youtu.be):Geoffrey Hinton talk,  2017. 4. 3., 1:11

- [Capsule Networks: An Improvement to Convolutional Networks](https://www.youtube.com/watch?v=VKoLGnq15RM): Siraj Raval, 22min

- [Capsule Networks (CapsNets) – Tutorial](https://www.youtube.com/watch?v=pPN8d0E3900&feature=youtu.be)

## 6. Material (Pdf, ppt)

- [Does the Brain do Inverse Graphics?](http://cseweb.ucsd.edu/~gary/cs200/s12/Hinton.pdf): pdf, Geoffrey Hinton, University of Toronto

- [Introduction to Capsule Networks (CapsNets)](https://www.slideshare.net/aureliengeron/introduction-to-capsule-networks-capsnets): ppt 56, 

## 7. Implementation (Project)

- [TF#1](https://github.com/debarko/CapsNet-Tensorflow), [TF#2](https://github.com/naturomics/CapsNet-Tensorflow)

- [Keras](https://github.com/XifengGuo/CapsNet-Keras)

- [PyTorch](https://github.com/nishnik/CapsNet-PyTorch)

## 8. Research Group / Conference 

---
```
 Capsule Network! 네트웍에 레이어를 추가 하는 대신 nested 하게 넷트웍을 쌓는 아이디어 입니다. 이 새로운 구조가 CNN을 밀어낼수 있을까요?
```

```
Hinton의 CapsNet에 대한 쉬운 설명(CNN과 비교)과 구현
CNN은 본질적으로 많은 뉴런을 쌓는 시스템입니다. 이러한 네트워크는 이미지 분류 문제를 처리할 때 탁월한 것으로 입증되었습니다. 그러나 계산상으로 정말 비싸기 때문에 이미지의 모든 픽셀을 신경망으로 맵핑하는 것은 어려울 것입니다. 따라서 convolution은 데이터의 본질을 잃지 않고 계산을 단순화하는 데 도움이되는 방법입니다. convolution은 기본적으로 많은 행렬 곱셈과 그 결과의 합계입니다.
CNN은 데이터세트에 매우 근접한 이미지를 분류할 때 탁월한 성능을 발휘합니다. 이미지가 약간 회전, 기울어 지거나 다른 방향일 경우 CNN 자체로 감지를 처리 ​할 수 ​​없습니다. 이 문제는 학습 도중 동일한 이미지의 다양한 변형을 추가하여 해결되었습니다.
Geoffrey Hinton은 인간의 뇌는 "캡슐 (capsule)"이라는 모듈을 가지고 있다고 주장합니다. 이 캡슐은 자세(위치, 크기, 방향), 변형, 속도, albedo, 색조, 텍스처 등 다양한 유형의 시각적 자극 및 인코딩을 처리할 때 특히 유용합니다. 뇌는 낮은 수준의 시각 정보를 "라우팅"하는 메커니즘을 가져야 합니다 이것이 처리를 위한 최고의 캡슐이라고 믿는 것입니다.
캡슐은 신경층의 중첩 세트입니다. 따라서 일반적인 신경망에서는 더 많은 레이어를 계속 추가합니다. CapsNet에서는 단일 레이어 안에 더 많은 레이어를 추가 할 수 있습니다. 다르게 설명하면 다른 내부에 신경층을 중첩시킵니다. 캡슐 내부 신경의 상태가 화상 안에 하나의 엔티티의 특성을 포착하고 캡슐은 엔티티의 존재를 나타내는 벡터를 출력합니다. 벡터의 방향은 엔티티의 속성을 나타냅니다. 벡터는 신경망의 모든 가능한 parent에게 전송됩니다. 가능한 모든 parent에 대해 캡슐은 Prediction vector를 찾을 수 있습니다. Prediction vector는 자신의 weight와 weight matrix를 곱하여 계산됩니다. 어느 parent가 가장 큰 스칼라 Prediction vector product가 캡슐 결합을 증가시킵니다. parent의 나머지는 그것들의 결합을 줄입니다. 이 routing by agreement는 max-pooling과 같은 현재 메커니즘보다 우수합니다. Max pooling route는 하위 계층에서 탐지된 가장 강력한 피처를 기반으로 합니다. dynamic routing 외에도 CapsNet은 캡슐에 squashing 추가에 대해 말합니다. squashing은 비선형 입니다. 따라서 CNN에서와 같이 각 레이어에 squashing 처리를 추가하는 대신 중첩된 레이어 집합에 squashing 처리를 추가 할 수 있습니다. squashing 함수는 각 캡슐의 벡터 출력에 적용됩니다.
또한 이 논문에서는 새로운 squashing 함수도 제안합니다. ReLU 또는 유사한 비선형 함수는 단일 뉴런과 잘 작동합니다. 그러나 이 논문은 이 squashing 기능이 캡슐에서 가장 잘 작동함을 발견했습니다. 이것은 캡슐의 출력 벡터의 길이를 squashing하려고 시도합니다. 작은 벡터인 경우 0으로 채우고 벡터가 길면 출력 벡터를 1로 제한합니다. dynamic routing은 약간의 계산 비용을 추가합니다. 그러나 확실히 장점이 있습니다.
```

```
Hinton의 새로운 접근 방식 capsule network(CapsNet)
컴퓨터가 이미지나 비디오를 통해 세상을 더 잘 이해할 수 있도록 신경 네트워크를 변형.
최고의 인공지능 시스템과 평범한 유아 사이의 걸림돌을 줄이기 위한 Hinton의 아이디어는 컴퓨터 비전 소프트웨어에 대한 지식을 조금 더 구축하는 것입니다. 캡슐(작은 그룹의 거친 가상 뉴런)은 고양이의 코와 귀 같은 공간의 다른 부분을 추적하도록 설계되었습니다. 많은 캡슐로 구성된 네트워크는 새로운 장면이 실제로 이전에 본 무언가에 대한 다른 시각인지 이해하기 위한 인식을 사용할 수 있습니다.
캡슐은 뉴런 그룹으로, 개체 또는 개체 파트와 같은 특정 유형의 엔터티의 인스턴스화 파라미터를 나타냅니다. 엔티티가 존재할 확률과 인스턴스화 파라미터를 나타내는 방향을 나타내기 위해 activity vector의 길이를 사용합니다. Active capsules은 한 수준에서 transformation matrices를 통해 상위 수준 캡슐의 인스턴스화 파라미터에 대한 예측을 합니다. 여러 예측이 일치하면 더 높은 레벨의 캡슐이 활성화됩니다. 차별화된 훈련을 거친 멀티 레이어 캡슐 시스템이 MNIST에서 최첨단 성능을 달성하고 많이 겹치는 숫자를 인식할 때 convolutional net보다 훨씬 뛰어나다는 것을 보여줍니다. 이러한 결과를 얻으려면 routing-by-agreement 메커니즘을 사용합니다.
```
```
힐튼 교수님이 “인간의 뇌는 CNN처럼 동작하는 게 아니야!”라면서 내놓은 CapsNet의 케라스 버전입니다. 정방형의
사람 얼굴만 학습한 CNN은 얼굴이 기울어지는 등 변형이 일어나면 인식을 잘 하지 못합니다. 이를 해결하기 위해 다양한 상황을 연출할 수 있는 데이터 부풀리기(data augmentation)기법을 사용합니다만 우리 뇌는 이렇게 무식하게 학습하지 않죠. 하나의 사진만 잘 학습하더라도 회전, 밝기의 변화 등이 일어나도 곧 잘 인지합니다. 이렇게 사물의 본질과 변형에 대해 추론할 수 있는 것을 “캡슐”이라 부르고 이를 구현하기 위에 레이어 안에 레이어를 중첩했다고 합니다. 이제 무식하게 데이터 부풀리기를 안해도 되는 날이 올까요? 이 CapsNet이 CNN을 대체할꺼라고 많은 분들이 기대하고 있는 것 같습니다. 케라스 유저분들께서는 일단 다운받고 돌려보세요~ 결과를 공유해서 같이 스터디 했으면 좋겠습니다.
```