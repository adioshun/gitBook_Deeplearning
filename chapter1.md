# 기초학습

## 1. Prediction
### 1-1 One Variable(리니어) Regression 
* Hypothesis : 데이터에 잘 맞는 직선은 무엇인가? 
    * $$H(x) = wx+b$$
* cost(loss) 함수 : 가설이 실제값에 얼마나 맞는가 검사 
    * (가설값-실제값)의 제곱의 합 평균
* 목적 : Cost함수를 최소화 하는 w,b 찾기 

##### Gradient Decent 
* Minimize Cost Function 
* W 값에 따른 Cost(w) 그래프를 그리고 최소 값 찾기
* 최소값 찾는 방법 = W,b값을 조금씩 바꾸어 가면서 경사도(=기울기)를 기반으로 0에 가까운값 찾기
* 경사도 구하는 법 = 미분 활용 
![](/assets/re_deep.PNG)
* x = W, y  = Cost(w)
* $$ \alpha $$ = Learning rate, 기울기를 찾아 내려 갈때 이동거리(발자국)

* $$ \frac{\partial}{\partial W} cost(w) $$ : 코스트 함수를 미분


### 1-2 Multi Variable Regression
* Hypthesis : 변수 수만 변함
    * $$H(x_1,x_2,x_3) = w_1x_1 + w_2x_2 + w_3x_2 +b  $$
* cost(loss) 함수 : 변화 없음

> Variable이 많을 경우 계산시 Matrix(행렬)연산 사용

![](/assets/multimatmal.PNG)
![](/assets/Screenshot from 2017-02-18 05-50-14.png)
* Bias를 행렬에 포함시키기 위해 w의 앞에 b를 x의 위에 1을 입력

## 2. Classification  

### 2-1 Bi-Classification : Logistic Regression 
* 딥러닝에 많이 사용되므로 중요
    * 실수를 On or Off로 구분 하기 위해서 
* Hypothesis : 리니어 H + 시그모이드 G 혼합 
    * $$ H(x) = \frac{1}{1+e^-WX}    $$

> 혼합이유 : 0~1의 값을 내어야 함 = 시그모이드(=로지스틱)함수 이용 하여 가능 = $$ g(x) = \frac{1}{1+e^-z}    $$

* cost(loss) 함수 
수식 설명 https://youtu.be/6vzchGYEJBc?t=4m48s


### 2-2 Multi-classification : Softmax Classification
* 여러 로지스틱을 가지고도 구현 가능하지만 추가 기능 필요
    * eg. A,B,C학점 = A or Not, B or Not, C or Not

![](/assets/Screenshot from 2017-02-18 06-26-32.png)
2.0 / 1.0 / 0.1의 결과가 아니라 총합이 1인 확률값(0.7/0.2/0.1)으로 나오게 하는것 
>one-hot Encoding을 이용하면 가장 큰값만 선택하고 나머지는 0으로 선택(argmaxg함수 사용)

* cost(loss) 함수 : Cross-Entropy
* Cost 함수(Cross Entropy)설명 :[Youtube](https://youtu.be/jMU9G5WEtBc?t=4m9s) 