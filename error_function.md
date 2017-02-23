# 오차 함수와 출력층의 설계

|문제의 유형|활성화 함수|오차함수|
|-|-|-|
|회귀|항등사항|제곱오차식|
|이진분류|로지스틱 함수||
|다클래스|소프트맥스함수|교차엔트로피|


## 1. 회귀 
### 1.1 정의 
* 출력값이 연속값을 같는 함수 대상

### 1.2 활성화 함수
* 목적으로 하는 함수와 같은 치역을 갖는 함수를 출력층의 활성화 함수로 설정 해야 함
    * 목표함수가 [-1"1]= 쌍곡선 정접함수
    * 목표함수가 임의의 실수($$ -\infty:\infty $$) = 항등사상


### 1.3 오차함수 
* 제곱오차 사용
$$
E(w) = \frac{1}{2} \sum_{n=1}^n\parallel d-y(x;w)\parallel^2
$$

* $$ \frac{1}{2} $$를 하는 이유는 미분시 2가 곱해지는걸 상쇄 하기 위함 



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



