# 리니어 리그레이션 
* Hypothesis : 데이터에 잘 맞는 직선은 무엇인가? 
    * H(x) = wx+b
* cost(loss) 함수 : 가설이 실제값에 얼마나 맞는가 검사 
    * (가설값-실제값)의 제곱의 합 평균
* 목적 : Cost함수를 최소화 하는 w,b 찾기 

# Gradient Decent 
* Minimize Cost Function 
* W 값에 따른 Cost(w) 그래프를 그리고 최소 값 찾기
* 최소값 찾는 방법 = W,b값을 조금씩 바꾸어 가면서 경사도(=기울기)를 기반으로 0에 가까운값 찾기
* 경사도 구하는 법 = 미분 활용 
![](/assets/re_deep.PNG)
* x = W, y  = Cost(w)
* $$ \alpha $$ = Learning rate, 기울기를 찾아 내려 갈때 이동거리(발자국)

* $$ \frac{\partial}{\partial W} cost(w) $$ : 코스트 함수를 미분



# 머신러닝과 행렬 연산

![](/assets/multimatmal.PNG)
https://youtu.be/MFAnsx1y9ZI?t=7m38s