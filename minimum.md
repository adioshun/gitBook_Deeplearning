# 오차 최소화 
FeedForward 신경망의 학습은 주어진 훈련데이터로 부터 계산되는 오차함수를 신경망의 파라미터(가중치, 바이어스) w에 대하여 `최소화` 하는 과정

## 1. 경사 하강법
* 비선형함수의 최소화 방법 중 가장 단순한 방법

> 모멘텀(Momentum) : 경사 하강법 성능 향상 기법 

> * 가중치의 업데이트 값에 이전 업데이트 값의 일정 비율을 거하는 방법 

> 참고 : [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/index.html), [arXiv](https://arxiv.org/abs/1609.04747)

## 2. 뉴턴법
* 목적함수의 2차 미분을 이용 
* 속도가 느림 

## 3. 확률적 경사 하강법 
* Stochastic gradient descent(SGD)
* 샘플을 다르게 하여 w를 학습
* 장점: 속도가 빠르다. 온라인 학습[^1]에 적용가능 
* [참고:Stochastic Gradient Descent 자동 학습속도 조절 알고리즘 정리](http://keunwoochoi.blogspot.com/2016/12/stochastic-gradient-descent.html)



---
[^1]: 훈련데이터의 수집과 최적화가 동시에 진행되는 방식 

[simple_gradient_descent.py](https://gist.github.com/chris-chris/808d383f19f74c537d9b4476b019c59a)
[Gradient Descent Overview](https://brunch.co.kr/@chris-song/50)
[참고 : 계산 그래프로 역전파 이해하기](https://brunch.co.kr/@chris-song/22)