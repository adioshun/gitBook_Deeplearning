we  propose  two  novel constrains in the design of deep structure to guarantee the performance gain when going deep

> 깊은 네트워크를 구성하여도 성능을 보장하는 2가지 제약에 대하여 제안한다.

1. Firstly, for each convolutional layer, its capacity of learning more complex patterns should be guaranteed

2. Secondly, the receptive field of the topmost layer should be no larger than the image region

위 두가지 제안 제약을 통해 다음이 가능하다. : we cast the task of designing deep convolutional neural network into a constrained opti-mization problem


# 1. Introduction 

Going deep greatly improves the **learning/fitting** capacity of the entire network while only increase model size and computational cost linearly 

>네트워크를 깊게 하면 **learning/fitting**이 증가 한다.

it is still unclear how to design a very deep convolution neural network effectively 

> 하지만, 효율적으로 깊게 설계 하는 법에 대하여서는 아직 잘 알려진 바가 없다. 

In this work, we propose a practical theory for designing very deep convolutional neural network effectively.

> 본 논문에서는 효육적으로 깊은 CNN을 설계 하는 방법을 제안 하려 한다. 

본 논문에서는 CNN을 두 Level로 나누었다. Classifier Level은 거의 비슷하므로, Feature Level에 대하여 주로 다루겠다. 

|Classifier Level |Feature Level|
|-|-|
|small feature map | 본 논문에서 다룸 |
|Large Conv Kernel||
|Identical for All Nets||

We cast(간주하다??)  the design of deep convolutional neural network into a constrained optimization problem.

The objective is __maximizing the depth of the target convolutional neural network__, subjecting to two constraints: 

1. the c-value of each layer should not be too small
    - c-value is a metric for measuring the capacity of learning more complex patterns; 

2. the receptive field of the topmost convolutional layer in the feature-level should no larger than image size.

> CNN은 설계는 깊이는 최대화 하면서 2가지 제약을 만족(subjecting)하는 **constrained optimization problem**과 같다. 

It is worth noting that our theory is task independent. The architecture of the designed convolutional neural network only depends on the raw image size and the fixed filter size.

> 제안 이론은 Task Independent하다는 점에서 주목할만 하다. Dependent는 `raw 이미지 크기`와 고정된 `필터 크기` 뿐이다. 

그 증거는 최근의 VGG, GooleNet이 여러 Task에 의존없이 좋은 성능을 보이는 것이다. (제안 이론과 비슷한 하므로)


Of course, we believe better architectures can be designed by incorporating task specific knowledge, such as the `cyclic pooling` and `rolling network`

> 물론 incorporating task specific knowledge로 설계된 아키텍쳐는 더 좋은 성과를 낼것이다. eg. `cyclic pooling` and `rolling network`

