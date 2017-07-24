# Hyper Parameter (하이퍼파라미터) 최적화 

하이퍼파라미터: 뉴런수, 배치크기, 학습률, 가중치 감소 

![](http://i.imgur.com/1LC5xiV.png)
- Train 데이터 : 학습시 사용
- Test 데이터 : 범용성능 평가
- Validation 데이터 : `하이퍼파라미터` 성능평가

> Test 데이터로 범용성능, 파라미터 성능 평가를 동시에 하면 안된다. 

---

## 1. 하이퍼파라미터 종류 

### 1.1 Learning rate

- 학습 진도율은 “gradient”의 방향으로 얼마나 빠르게 이동을 할 것인지를 결정 한다.

- 학습 진도율이 너무 작으면 학습의 속도가 너무 느리게 되고, 반대로 너무 크면 학습의 결과가 수렴이 안되고 진동을 하게 될 수도 있다.

### 1.2. Cost function

- 일반적으로 최소 자승법 or Cross-entropy 함수 사용


### 1.3.Regularization parameter

 - Overfitting의 문제를 피하기 위해, L1 또는 L2 regularization 방법을 사용
 
 - 일반화 변수(λ)는 weight decay의 속도를 조절하기 위한 용도로 사용할 수가 있다.

 
### 1.4.Mini-batch 크기

- Mini-batch의 크기가 큰 경우는 병렬연산 구조를 사용할 때 효과적일 수 있으며, 크기가 작으면 더 많은 update를 할 수가 있다.

### 1.5.Training 반복 횟수

- 학습의 조기 종료(early stopping)을 결정하는 변수가 된다.

- Early-stopping이란 validation set을 이용해서 학습의 효율이 더 이상 올라가지 않게 되면, 조기에 학습을 종료하는 것을 말하며, overfitting을 방지할 때 중요하게 사용된다.

### 1.6.Hidden unit의 개수

- hidden layer의 개수가 많아질수록 특정 훈련 데이터에 더 최적화 시킬 수가 있다.

- 또한 모든 hidden layer의 뉴런의 개수를 동일하게 유지하는 것이 같은 hidden layer의 개수에 뉴런의 개수를 가변적으로 하는 것보다 효과적이다.

- 또한 첫번째 hidden layer에 있는 뉴런의 개수가 input layer에 있는 뉴런의 개수보다 큰 것이 효과적인 경우가 많다.

 

### 1.7. 가중치 초기화(Weight initialization)

- 바이어스는 일반적으로 0으로 초기화가 많이 된다.
 - 하지만 가중치의 경우는 초기화가 학습 결과에 큰 영향을 끼치기 때문에 주의가 필요하다.


- 가중치는 보통 무작위로 초기화가 되며 범위는 [-r, r] 범위를 가진다.
 - r은 input layer에 있는 뉴런의 개수 제곱의 역수가 된다. 
 - eg. 가령 입력 뉴런의 개수가 6이라면, [-1/36, 1/36] 범위 내에서 무작위로 설정을 한다.


#### 1.8 Filter의 수 (CNN 한정) 

$$

T_c = N_p \times N_f \times T_k 

$$

- $$T_c$$: 각 layer에서의 연산시간,
- $$N_p$$: 출력 pixel의 수
- $$N_f$$: 전체feature map의 개수
- $$T_k$$: 각 filter 당 연산 시간

각 단에서의 연산 시간/량을 비교적 일정하게 유지하여 시스템의 균형을 맞추는 것
- 각 layer에서 feature map의 개수와 pixel 수의 곱을 대략적으로 일정하게 유지시킨다.
 - 그 이유는 위 식에서 살펴본 것처럼 convolutional layer에서의 연산 시간이 픽셀의 수와 feature map의 개수에 따라 결정이 되기 때문이다.
 - 보통 pooling layer를 거치면서 2x2 sub-sampling을 하게 되는데, 이렇게 되면 convolutional layer 단을 지날 때마다, pixel의 수가 1/4로 줄어들기 때문에 feature map의 개수는 대략 4배 정도로 증가시키면 될 것이라는 것은 감을 잡을 수가 있을 것이다.
 - Feature map의 개수는 가중치와 바이어스와 같은 free parameter의 개수를 결정하며, 학습 샘플의 개수 및 수행하고자 하는 과제의 복잡도에 따라 결정이 된다.
 


#### 1.8 Filter의 형태 (CNN 한정) 

학습에 사용하는 학습 데이터 집합에 따라 적절하게 선택
- 작은 크기(32x32나 28x28과)의 입력 영상 :  5x5 필터
- 큰 크기 입력 OR 1 단계 필터 : 11x11 OR 15x15 필터 
- 3단계 이상 : 3x3 필터 (Krizhevsky의 논문)

큰 kernel 크기를 갖는 필터 1개 Vs. 작은 크기를 갖는 filter 여러 개
- 여러 개의 작은 크기의 필터를 중첩해서 사용하는 것이 좋다


#### 1.9 Stride 값 (CNN 한정) 

Stride는 convolution을 수행할 때, 건너 뛸 픽셀의 개수
- LeCun의 논문은 stride를 1을 사용했지만,
- Krizhevsky의 논문에서는 1단계 convolution에서는 stride 값을 4를 사용

Stride는 입력 영상의 크기가 큰 경우, 연산량을 줄이기 위한 목적으로 입력단과 가까운 쪽에만 적용

통상적으로 보았을 때는 stride를 1로 하고, pooling을 통해 적절한 sub-sampling 과정을 거치는 것이 결과가 좋다

> 출처 : [라온피플 블로그](http://laonple.blog.me/220571820368)

#### 1.10 Zero-padding 지원 여부 (CNN 한정)

아래의 장점으로 인하여 사용 하는것이 좋음 
 - 영상의 크기를 동일하게 유지
 - 경계면의 정보까지 살릴 수가 있음
 
 


---
## 2. 최적화 방법 

### 2.1 Manual Search(직관에 의한 방법 )

- 반복 수행을 통해서 범위를 줄여 나감.  

- 결과를 판정하기 위한 validation set가 필요하다.

### 2.2 Grid Search

- 큰 틀에서 보면, Manual search와 큰 차이가 없으며, 개념적으로도 비슷하다.

- 단, Grid search의 경우는 선험적인 지식을 활용하여 문제를 분석하고, hyperparameter의 범위를 정한다.
 - 그리고 그 범위 안에서 일정한 간격으로 점을 정하고 
 - 그 점들에 대해 1개씩 차례로 실험을 해보면서 최적의 값을 찾은 후
 - 다시 best로 추정이 되는 점을 기준으로 세분화하여 최적값을 찾는 방법이다


- 결과를 판정하기 위한 validation set가 필요하다.



### 2.3 Random search

- Grid search와 마찬가지로 선험적인 지식을 이용하여 hyperparameter의 범위를 정한다.

- 무작위로 최적값을 찾는 작업을 진행을 한다.(Grid는 일정한 간격으로 탐색) 
 -  ‘유한 자원’을 기반으로 해야 할경우 사용 


### 2.4 베이즈 최적화 

- Bayesian optimization의 기본 원리가 prior knowledge를 활용하는데 있으므로,

- 현재까지의 실험 결과를 바탕으로 통계적인 모델을 만들고,

- 그것을 바탕으로 다음 탐색을 해야 할 방향을 효과적으로 정하자는 것이 이 방법의 핵심이다.

> 참고 : Practical bayesian optimization of machine learning algorithm





---

###### [참고] epchos(에포크)
에포크는 모의고사 1회분을 몇 번 풀어볼까 입니다. 즉 100문항의 문제들을 몇 번이나 반복해서 풀어보는 지 정하는 것입니다. 에포크가 20이면 모의고사 1회분을 20번 푸는 것입니다. 처음에는 같은 문제를 반복적으로 풀어보는 것이 무슨 효과가 있는 지 의문이 들었지만 우리가 같은 문제집을 여러 번 풀면서 점차 학습되듯이 모델도 같은 데이터셋으로 반복적으로 가중치를 갱신하면서 모델이 학습됩니다. 같은 문제라도 이전에 풀었을 때랑 지금 풀었을 때랑 학습상태(가중치)가 다르기 때문에 다시 학습이 일어납니다.

> 모의고사 1회분을 20번 푸는 것과 서로 다른 모의고사 20회분을 1번 푸는 것과는 어떤 차이가 있을까요? 데이터 속성에 따라 다름 
  