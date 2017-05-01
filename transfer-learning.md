# Transfer Learning

만약 이미지셋이 100만개보다 적다면 pre-train 모델을 사용하라!!!

![](http://sebastianruder.com/content/images/2017/03/andrew_ng_drivers_ml_success-1.png)

> "Transfer learning will be the next driver of ML success", Andrew NG, NIPS 2016


## 역사??
- 1. 1995년 NIPS workshop에서 "Learning to learn"이라는 주제로 소개
- 2. Domain Adaptation : knowledge transfer
- 3. Transfer learning

<img id="image_canv" src="http://nmhkahn.github.io/assets/CNN-Practice/vggnet.png" rotate="70">

![](http://nmhkahn.github.io/assets/CNN-Practice/vggnet.png)

위 이미지와 같은 VGGNet을 어떤 도메인에 사용해야 할 경우가 있다고 생각해보자. 맨땅에 VGGNet을 학습시키기 위해서는 매우매우 많은 데이터가 필요하고, 학습하는데 걸리는 시간도 매우 긴 것은 사실이다 (2~3주 걸린 것으로 알고 있다). 하지만 VGGNet의 pre-train된 모델을 구할 수 있다면 (Caffe model zoo) 이를 갖다가 사용하면 문제가 매우 쉬워진다.

만약 적용해야 하는 도메인의 데이터셋이 작다면 VGGNet의 끝에 있는 FC-1000과 softmax 레이어를 지워버리고 내게 맞는 도메인 예를 들어 CIFAR-10이라면 FC-10, softmax로 바꾼다. 또 softmax 대신 SVM을 사용하거나 다른 loss function을 사용하거나.. 뭐 자기 마음이다. 그리고 네트워크를 학습 할 때 기존의 레이어는 pre-train된 모델의 weight, bias를 그대로 사용하고 추가한 레이어만 학습을 시킨다. 모든 레이어를 학습시키면 좋은 성능이 나올지는 모르지만 적은 데이터셋에서는 그렇게 효과적인지 의문이고 가장 큰 문제는 학습이 오래걸린다..

도메인의 데이터셋이 적당히 있다면 추가한 레이어와 그 위 몇개 레이어를 (그 위 maxpool과 conv 2개 정도?) fine-tune한다. 데이터가 많으면 많을 수록 fine-tune하는 레이어를 늘려도 괜찮고 데이터가 꽤 많으면 모든 레이어를 학습시키는 것도 고려할 수 있다.
fine-tune을 할 때 한가지 팁은 새로 추가한 레이어의 learning_rate는 기존 네트워크에서 사용한 것 보다 10배 작게, 중간 레이어들은 100배 작게 learning_rate를 사용하는 게 좋다고 한다.

정리하자면 다음과 같다.

- 도메인이 기존 데이터셋과 비슷하고, 데이터가 적다 : 끝 레이어(top layer)에 도메인에 맞는 레이어를 추가하고 추가한 레이어만 학습한다.
- 도메인이 기존 데이터셋과 비슷하고, 데이터가 많다 : 추가한 레이어와 몇개 레이어를 fine-tune 한다.
- 도메인이 기존 데이터셋과 매우 다르고, 데이터가 적다 : 큰일이다…..
- 도메인이 기존 데이터셋과 매우 다르고, 데이터가 많다 : 많은 레이어를 fine-tune 한다.

뉴럴넷의 얕은 레이어(입력 이미지와 가까운 레이어)는 edge나 texture를 검출하는 등의 역할을 하는 이미지에 대해 매우 포괄적으로 사용 가능한 레이어이다. 반면에 깊은 레이어는 학습에 사용된 데이터셋에 specfic하기 때문에 얕은 레이어도 fine-tune하면 물론 좋지만, 꼭 그럴 필요는 없다.


> 출처 : http://nmhkahn.github.io/CNN-Practice


####  PathNet
인공지능은 전이학습을 통해 이전의 학습에서 배운 지식을 완전히 새로운 과제에 활용할 수 있음.

- 이전의 지식을 활용함으로써 인공지능은 새로운 과제에 새로이 신경망을 구축할 필요가 없으며, 전이학습은 과제 이행의 질과 시간적 효율성을 높여줌.

전이학습은 신경망을 구축하고 학습하는데 있어서 가장 큰 과제이며, DeepMind는 PathNet을 통해 이를 해결하기 위해 노력중임.
- PathNet이란 SGD(stochastic gradient descent)와 유전 선발방법(genetic selection method)을 사용하여 학습한 신경망 네트워크를 뜻함.
- PathNet은 여러 층(layer)의 모듈로 이루어져 있으며, 각각의 모듈은 다양한 형태의 신경망임(예: 나선형(convolutional), 피드포워드(feedforward) 또는 반복되는(recurrent) 형태).

학습을 하기에 앞서, 모듈에 대한 정의를 해야 함(L = 층수(number of layers), N = 하나의 층이 가질 수 있는 최대 모듈수).
- 마지막 층은 복잡하게 이루어져 있으며, 다른 과제와 함께 공유되지 않음.

- A3C(Advantage Actor-critic)를 활용하는 마지막 층은 가치함수(value function)와 정책평가(policy evaluation)을 나타냄.

- 모듈에 대한 정의가 이루어진 후, P 유전자형(또는 경로)가 네트워크에 생성됨. 각각의 유전자형 평가를 통해 최적화된 경로를 파악하고, 이 경로를 통해 학습을 힘.

- 경로의 학습은 백 프로퍼게이션(back propagation)을 사용한 SGD(stochastic gradient descent)를 통해 이루어지며, 한 번에 하나의 경로를 통해 실행됨.

과제를 학습한 후 네트워크는 최적 경로의 모든 매개변수(parameters)를 수정하며, 모든 매개변수는 다시 초기화 됨.

- A3C는 이전지식이 지워지지 않도록 이전과제의 최적경로가 수정되지 않도록 함.

> 출처 : [미래금융연구센터](http://www.kcft.or.kr/2017/02/2120)

---
[A Survey on Transfer Learning - SJ Pan et al. 2010](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)