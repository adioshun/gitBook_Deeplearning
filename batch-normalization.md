# Batch normalization

> 출처 : [KIM BEOMSU 블로그](https://shuuki4.wordpress.com/2016/01/13/batch-normalization-설명-및-구현/)
> 참고 : [번역_Batch Normalization (ICML 2015)](http://sanghyukchun.github.io/88/)

![](http://spawk.fish/images/posts/2016-02/policy-network/adapter.png)

## 1. 개요 
각층의 활성화를 적당히 퍼트리도록 강제하는 기법 

- 장점1 : 학습 속도 개선
- 장점2 : 초기값에 크게 의존하지 않는다. (초기값 선택의 문제 해결)
- 장점3 : 오버피팅을 억제 (드랍아웃등의 필요성 감소)

> 보통 활성함수 앞에 위치, 활성함수 뒤에 위치하는 방법도 연구중 

* 기본적으로 Gradient Vanishing / Gradient Exploding 이 일어나지 않도록 하는 아이디어 중의 하나이다.
* 기존 해결법
    * Activation 함수의 변화 (ReLU 등)
    * Careful Initialization
    * small learning rate 등


###### 목적 
* 위의 기존의 간접적인 해결법 보다는 training 하는 과정 자체를 전체적으로 안정화하여 학습 속도를 가속시킬 수 있는 근본적인 방법을 찾고싶어 했다
* 불안정화가 일어 나는 이유 : Internal Covariance Shift[^1]
* 불안정화(=Internal Covariance Shift) 해결 방법 : 각 층의 input의 distribution을 평균 0, 표준편차 1인 input으로 normalize 시키는 방법 -> Whitening[^2]으로 해결 가능 
* Whitening의 문제점
    * whitening은 계산량이 많음(covariance matrix, inverse 계산이 필요) 
    * whitening을 하면 일부 parameter 들의 영향이 무시된다

## 1. 논문의 내용 
whitening의 단점을 보완하고, internal covariance shift는 줄이기 위해 논문에서는 다음과 같은 접근을 취했다.
*  각각의 feature들이 이미 uncorrelated 되어있다고 가정하고, feature 각각에 대해서만 scalar 형태로 mean과 variance를 구하고 각각 normalize 한다.
* 단순히 mean과 variance를 0, 1로 고정시키는 것은 오히려 Activation function의 nonlinearity를 없앨 수 있다. 예를 들어 sigmoid activation의 입력이 평균 0, 분산 1이라면 출력 부분은 곡선보다는 직선 형태에 가까울 것이다. 또한, feature가 uncorrelated 되어있다는 가정에 의해 네트워크가 표현할 수 있는 것이 제한될 수 있다. 이 점들을 보완하기 위해, normalize된 값들에 scale factor (gamma)와 shift factor (beta)를 더해주고 이 변수들을 back-prop 과정에서 같이 train 시켜준다.
* training data 전체에 대해 mean과 variance를 구하는 것이 아니라, mini-batch 단위로 접근하여 계산한다. 현재 택한 mini-batch 안에서만 mean과 variance를 구해서, 이 값을 이용해서 normalize 한다.


## 2. 알고리즘 
뉴럴넷을 학습시킬 때 보통 mini-batch 단위로 데이터를 가져와서 학습을 시키는데, 각 feature 별로 평균과 표준편차를 구해준 다음 normalize 해주고, scale factor와 shift factor를 이용하여 새로운 값을 만들어준다
![](https://shuuki4.files.wordpress.com/2016/01/bn1.png)
![](https://shuuki4.files.wordpress.com/2016/01/bn2.png)

## 3. CNN에 적용시 
CNN에 적용시키고 싶을 경우 지금까지 설명한 방법과는 다소 다른 방법을 이용해야만 한다. 

1. 먼저, convolution layer에서 보통 activation function에 값을 넣기 전 Wx+b 형태로 weight를 적용시키는데, Batch Normalization을 사용하고 싶을 경우 normalize 할 때 beta 값이 b의 역할을 대체할 수 있기 때문에 b를 없애준다. 
2. 또한, CNN의 경우 convolution의 성질을 유지시키고 싶기 때문에, 각 channel을 기준으로  각각의 Batch Normalization 변수들을 만든다. 
    * 예를 들어 m의 mini-batch-size, n의 channel size 를 가진 Convolution Layer에서 Batch Normalization을 적용시킨다고 해보자. 
    * convolution을 적용한 후의 feature map의 사이즈가 p x q 일 경우, 각 채널에 대해 m x p x q 개의 각각의 스칼라 값에 대해 mean과 variance를 구하는 것이다. 
3. 최종적으로 gamma와 beta는 각 채널에 대해 한개씩 해서 총 n개의 독립적인 Batch Normalization 변수들이 생기게 된다.

## 4. 장점 
* 기존 Deep Network에서는 learning rate를 너무 높게 잡을 경우 gradient가 explode/vanish 하거나, 나쁜 local minima에 빠지는 문제가 있었다. 이는 parameter들의 scale 때문인데, Batch Normalization을 사용할 경우 propagation 할 때 parameter의 scale에 영향을 받지 않게 된다. 따라서, learning rate를 크게 잡을 수 있게 되고 이는 빠른 학습을 가능케 한다.
* Batch Normalization의 경우 자체적인 regularization 효과가 있다. 이는 기존에 사용하던 weight regularization term 등을 제외할 수 있게 하며, 나아가 Dropout을 제외할 수 있게 한다 (Dropout의 효과와 Batch Normalization의 효과가 같기 때문.) . Dropout의 경우 효과는 좋지만 학습 속도가 다소 느려진다는 단점이 있는데, 이를 제거함으로서 학습 속도도 향상된다.
---
> 작년에 뉴럴넷 레이어 수를 처음으로 5개 이상으로 시도 해 봤을때 학습이 진행이 잘 안 되거나 에러가 뜨고 죽어 버리는 문제 때문에 한참 고민한 적이 있었다.
그때 Batch Normalization(S. Ioffe and C. Szegedy, 2015)을 알게 되었고 처음 이것(20줄 짜리 간단한 코드)을 적용 해 봤는데 학습 에러(Gradient 폭발)가 없어진 것은 물론이고 학습 결과(정확도)가 더 좋을 뿐 아니라 학습 속도가 10배 이상 빨라져서 짜릿했던 기억이 아직 생생하다.
게다가 이런저런 Weight Initialization 방법들 및 Dropout 등의 Regularization 방법들도 거의 다 필요 없어져 버려서 정말 편했었다.
한마디로 나에게 BN은 Silver Bullet이었다.
그런데 엊그제 Batch Renormalization(S. Ioffe, 2017)이란 것이 또 나왔다.
기존의 BN은 Mini Batch 크기가 작거나 그 샘플링 방법이 좀 어긋나면 효과적으로 동작 하지 않는다. 나도 그래서 고민이 많다. 그래픽 카드 메모리의 한계 때문에 Mini Batch 크기를 마음껏 늘릴수가 없고, 샘플링은 뭘 어떻게 해야 잘 하는 건지 모르겠다.
그런데 이번에 나온 이 Batch Renormalization은 (역시나 간단한 원리로...) 그러한 문제점을 개선 했다고 한다.
TF 구현체도 며칠 내로 나오겠지? 얼른 적용해 보고 싶다. 또 한 번 그 짜릿함을 느낄수 있으려나.
https://arxiv.org/abs/1702.03275
@ 그런데 S. Ioffe 이 분은 정말... 어떻게 이렇게 대단한 발견(또는 발명)을 하고 그걸 또 본인이 남들보다 먼저 발전 시킬수 있는 걸까... 그 실력과 에너지가 엄청나네 정말. 연구 하는 모습을 옆에서 보고 싶네.

---

1. [Batch Normalization : Accelerating Deep Network Training by Reducing Internal Covariance Shift, 2015](http://arxiv.org/abs/1502.03167)
2. [Batch normalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models,2017](https://arxiv.org/abs/1702.03275)
3. [Batch Normalization 설명 및 구현](https://shuuki4.wordpress.com/2016/01/13/batch-normalization-설명-및-구현/)


---
[^1]: Internal Covariance Shift: Network의 각 층이나 Activation 마다 input의 distribution이 달라지는 현상
[^2]: Whitening : 기본적으로 들어오는 input의 feature들을 uncorrelated 하게 만들어주고, 각각의 variance를 1로 만들어주는 작업 













