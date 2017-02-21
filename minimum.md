# 오차 최소화 
FeedForward 신경망의 학습은 주어진 훈련데이터로 부터 계산되는 오차함수를 신경망의 파라미터(가중치, 바이어스) w에 대하여 `최소화` 하는 과정

## 1. 경사 하강법
* 비선형함수의 최소화 방법 중 가장 단순한 방법

> 모멘텀(Momentum) : 경사 하강법 성능 향상 기법 
> * 가중치의 업데이트 값에 이전 업데이트 값의 일정 비율을 거하는 방법 
> * 

## 2. 뉴턴법
* 목적함수의 2차 미분을 이용 

## 3. 확률적 경사 하강법 
* Stochastic gradient descent(SGD)
* 샘플을 다르게 하여 w를 학습
* [참고:Stochastic Gradient Descent 자동 학습속도 조절 알고리즘 정리](http://keunwoochoi.blogspot.com/2016/12/stochastic-gradient-descent.html)



