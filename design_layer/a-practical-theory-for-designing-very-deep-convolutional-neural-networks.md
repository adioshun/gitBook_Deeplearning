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

## The design of classifier level

- 기존 방법 : The common design in classifier level is two fully connected layers with dropout on each of them. 
    - However we found this design is prone to over-fitting if the training set is not sufficiently large.

> 기존 방법 : two fully connected layers + dropout를 이용하는 방법이다. 하지만, 이 방법은 데이터가 적으면 오버피팅되는 경향이 있다. 

- 제안 방법 :  it is better to...
    1. down sample the input feature map to a small size (6x6, 7x7 or 8x8),
    2. and then apply two 5x5 convolutional layers, 
    3. finally max pool the feature map into a vector and dropout this vector.

As kernel size is very large relative to the feature map,herein the convolutional layers are more like fully connected layers. 
> 제안 방법은 피쳐맵에 비하여 커널 크기가 큰데, 이로인해서 conv 레이어는 일종의 FC 레이어처럼 된다. 

One perspective of understanding this design is it conduct dense sliding window test (used in VGG’s work) and aggregate the final results by max pooling.
> 이런 설계 방식은 VGG처럼 `Dense 슬라이딩 윈도우 테스트`를 수행하고 마지막 결과를 `max pooling`하여 합친다. 

The difference is that we found it is better to use large convolutional kernels relative to the feature map and replace the <del>average pooling</del> with `max pooling`.
> 제안 방식은 GoogleNet에서 영감을 얻었으며 기존 방법과 대비하여 다른점은 이 방식이 Feature map 대비 큰 conv커널을 사용하는데 좋다. Average pooling -> max Pooling을 사용하였다. 


# 2. Capacity of learning – the first constraint

The functionality of a convolutional layer is composing more complex patterns from input patterns.
> Conv레이어는 기능은 입력 패턴에선 더 복잡한 패턴을 Composing하는 것이다. 

![](http://i.imgur.com/7OUIlY5.png)
[학습률(Learning Capacity)를 표현한그림. Exemplar(a)가 (b)보다 좋은 효과를 보임. `파란박스`는 Conv filter/kernel]

In Figure 2(b), we show a simple case when a convolutional layer fail to learn a more complex pattern. In this case, the `filter size` is smaller than the `distance between the responses of pattern A and B`, 
- in other words, it can not detect both pattern A and B as well as their spatial relationship, therefore fail to learn the more complex pattern AB.

> 그림 2(b)는 complex pattern 학습에 실패 한것을 표현하고 있음. 이 경우 filter size가 패턴 A ~ B의 간격보다 작기 때문에 발생한다. 다시 말해 필터가 A와 B의 `패턴`과 `공간적 관계(spatial relationship)`특징을 탐지 하지 한것이다. 

To regain the capacity of learning, we can either use larger convolutional kernel, or cut the responses distance by half via down sampling with stride 2.
> 해결 방법은 (1) 더 큰 filter를 사용하거나, (2)Stride2로 다운 샘플링 하여 responses간의 거리를 반으로 줄이는 것이다. 


The odds(가능성) of the learning failure grows, as a convolutional neural network goes deep without down sampling, because the sizes of the detected patterns and their meaningful spatial relationships grows layer by layer.
> 다운 샘플링 없이 CNN을 깊게만 하면 학습실패 확률이 증가 한다. 왜냐 하면, `detected patterns`의 크기와 패턴의 공간적 관계(spatial relationship)`의 크기가 레이어가 깊어질수록 커지기 때문이다. 

Once the spatial relationships of the majority input patterns exceed the filter size of a convolutional layer, this convolutional layer will lose its capacity of learning more complex patterns.
> 입력 패턴 중에서 대부분의 공간적 관계(spatial relationship)가 filter size를 초과 하게 되면, 이 Conv 레이어는 학습 능력을 잃게 된다. 

학습 능력을 정적으로 측정하기 위해서 `c-value` 정의 하였다. 

$$

c-value = \frac{Real Filter Size}{Receptive Field Size}

$$
- Real filter size of a k-by-k convolutional layer is k if there is no down sampling, it **doubles** after each down sampling 
    - i.e. 2k after one down sampling and 4k after two down sampling etc. 

> Real filter size는 매 다운 샘플링 시 2배 증가 한다. 

- Receptive field size is defined as the `maximum size of a neuron` can see on the **raw image**.
    - It grows proportionally as the convolutional neural network goes deep. 

> Receptive field size는 이미지의 최대 뉴론 수이다. 

![](blob:http://imgur.com/9c83d30d-a161-4ad0-88e9-0f780461a2ba)

[Receptive field size 도식화/ size = `matlab-style arrays`] 

eg. 40: 8: 64 represents [40, 48, 56, 64]. 입력 이미지 크기는 64x64
- We subtract an annoying small constant ”1” when calculating the receptive field sizes in order to make the description and subsequent derivation more concise.

> [중요] spatial relationships는 측정이 불가능하기 때문에 대신 `receptive field size`를 사용하였음. 

|the first constraint 요약|
|-|
|the c-value of each convolutional layer should be larger than a minimum value t. We empirically found t = 1/6 is a good lower bound of c-value for all convolutional layers in various tasks.|
|`c-value`는 최소값 t(=1/6)보다 커야 한다|

