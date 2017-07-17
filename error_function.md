학습을 통해 직접적으로 줄이고자 하는 값을 손실(loss), 에러(error), 혹은 코스트(cost)라고 합니다.

학습을 통해 목표를 얼마나 잘(못) 달성했는지를 나타내는 값을 척도(metric)라고 합니다

http://deepestdocs.readthedocs.io/en/latest/002_deep_learning_part_1/0023/

# 오차 함수와 출력층의 설계

|문제의 유형|활성화 함수|오차함수|
|-|-|-|
|회귀|항등사항|제곱오차식 = $$E(w) = \frac{1}{2} \sum_{n=1}^n\parallel d-y(x;w)\parallel^2
$$|
|이진분류|로지스틱 함수|$$ E(w) = -\sum_{n=1}^{N}[d_n \log y(x_n;w) + (1+d_n) \log{1-y(x_n;w)}] $$|
|다클래스|소프트맥스함수|교차엔트로피 = $$ E(w) = -\sum_{n=1}^{N}\sum_{k=1}^{K} d_{nk} \log y_k(x_n;w) $$|
## 1. 회귀 
### 1.1 정의 
* 출력값이 연속값을 같는 함수 대상

### 1.2 활성화 함수
* 목적으로 하는 함수와 같은 치역을 갖는 함수를 출력층의 활성화 함수로 설정 해야 함
    * 목표함수가 [-1"1]= 쌍곡선 정접함수(=쌍곡선 탄젠트 함수)
    * 목표함수가 임의의 실수($$ -\infty:\infty $$) = 항등사상


### 1.3 오차함수 
제곱 오차 (Square Loss) = 평균 제곱 오차(mean-squared error, MSE)

$$
E(w) = \frac{1}{2} \sum_{k}(y_k - t_k)^2
$$
- $$y_k$$: 출력값(예상값)
- $$t_k$$: 정답(원래값)
- k = 데이터의 차원수 

> $$ \frac{1}{2} $$를 하는 이유는 미분시 2가 곱해지는걸 상쇄 하기 위함 


두 벡터간의 유클리드 거리(Euclidian distance)를 측정합니다. 

두 벡터가 가까워질수록 제곱차 손실은 줄어듭니다. 

제곱차 손실은 yy에 대한 볼록함수이기 때문에 손실 함수로 사용하기 좋습니다. 

단점 : 하지만 훈련 속도가 느리고 성능이 다른 손실 함수들보다 많이 떨어짐


## 2. 이진분류 
### 2.1 정의 
* 입력 x에 대하여 두 종류로 구별하는 문제 

### 2.2 활성화 함수
* 로지스틱 함수

### 2.3 오차함수 






* 선형관계에 놓여 있지 않은 일반화 선형모형에서는 `최대우도법`활용 
$$
E(w) = -\sum_{n=1}^{N}[d_n \log y(x_n;w) + (1+d_n) \log{1-y(x_n;w)}] 
$$

```
보통 모수와 모집단이 이미 알려져있고 여기서 어떤 현상이 관찰될 가능성을 확률이라고 하는데, 우도는 반대의 개념이다. 관측치가 고정되고, 그러한 관측치가 나오게 하는 가장 그럴 듯한 모수값을 추정하는 것이다. 이 때 '이 관측치가 관찰될 가능성'을 '우도'라고 하고, 함수로 표현하며, 우도가 가장 높아지게 하여 모수를 추정하는 방법이 최대우도법이다. 

```

## 3. 다클래스 
### 3.1 정의 
* 손글씨 분류등의 문제 (0~9까지 숫자)
* 출력층에 분류하려는 클래스 수(L)과 같은 수의 유닛을 구성 
### 3.2 활성화 함수
* 소프트맥스 함수 
    * 각 결과 확률값을 출력, 가장 높은 확률을 결과로 선택 
    * 출력 총합이 항상 1

### 3.3 오차함수 
* 교차 엔트로피(cross entropy, negative log likehood)
$$
E(w) = -\sum_{n=1}^{N}\sum_{k=1}^{K} d_{nk} \log y_k(x_n;w)
$$
* The cross-entropy is a performance measure used in classification. The cross-entropy is a continuous function that is always positive and if the predicted output of the model exactly matches the desired output then the cross-entropy equals zero. The goal of optimization is therefore to minimize the cross-entropy so it gets as close to zero as possible by changing the weights and biases of the model.`nn.softmax_cross_entropy_with_logits`

- [[동영상]참고_왜 크로스-엔트로피를 쓸까?](https://youtu.be/srdDQr07sGg): 머신/딥러닝에서 크로스-엔트로피를 코스트함수로 사용하는 이유 중 두개를 소개

## 4. 이진분류 + 다클래스 분류 : Entrpy Loss 이용 

### 4.1 Entropy Loss

- 기본적으로 Kullback-Leibler 발산(KL divergence)에 기반을 두고 있습니다. 
    - KL은 두 확률분포 사이의 거리(distance)를 측정합니다. 
    - 다이버전스:서로 다르게 움직이는 현상

- 즉, 두 확률분포가 얼마나 유사한지를 나타내 줍니다. 

###### Step 1. 이산확률변수(discrete random variable)의 경우에 대한 식은 아래와 같습니다. 

$$
D_{KL}(P,Q) = D_{KL}(P \parallel Q) = \sum_i p_i\log\frac{p_i}{q_i}

$$
- p,q는 확률분포: 모든 원소가 0 이상 1 이하의 값, 총 합은 1
    - p : 정답 확률분포
    - q : 예측 확률분포
- p,q가 완전히 동일(분포가 동일)하다면 D=0 
    - 목표 : 정답과 예측값의 확률 분포가 같도록 만들기

###### Step 2. p는 외부에서 주어지는 고정된 값(c)이므로 

$$
D_{KL}(P,Q) = \sum_i p_i\log\frac{p_i}{q_i} = \sum_i p_i\log p_i - \sum_i p_i\log q_i (로그의 뺼셈 성질)


$$
$$
= C- \sum_i p_i\log q_i 
$$

###### Step 3. 최종식 도출 (교차 엔트로피:cross entropy)
$$
L = - \sum_i p_i\log q_i   
$$

> 가능도(likelihood)를 통해 서도 식을 유추 할수 있어 `음수 로그 가능도(negative log-likelihood)`라도고 함 

### 4.2 Entropy Loss for Binary
- y가 0 혹은 1만 되는 경우(즉, 이진 분류 문제인 경우)

$$
L = - t\log y - (1-t) \log (1-y)
$$


### 4.3 Entropy Loss for Categorical 
- y가 여러 클래스를 갖는 경우

$$
L = -\sum_{i=1}^C t_i \log y_i
$$