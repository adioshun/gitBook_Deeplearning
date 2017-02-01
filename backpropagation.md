> How Can we learn $$w1, w2, b1, b2$$ from training data

Layer가 깊어 질수록 초기의 weight와 Bias를 찾기 힘들다 (Marvin Minsky,1969)
* Backpropagation으로 가능 (Paul1974, Hilton1986)

# Backpropagation
![](/assets/bp.PNG)
미분 chain rule 활용

1. $$ f(x) = wx + b$$를 도식화 한
2. 각 값이 $$f$$에 미치는 영향 구하기

## 1. 방법 A: Forward 
![](/assets/BP_1.PNG)
* 각 값을 사용 ($$w= -2, x=5, b=3$$)
* f = -7 계산 



## 2. 방법 B : Backward
* 기본 $$ f(x) = wx + b $$에서 미분, 편미분 통하여 값 도출
* $$ g= wx $$ 치환 하면 $$f(x) = g + b$$

###### 위 치환 식 $$ g= wx $$ 에서 
* w를 기준으로 g를 편미분하면 $$ \frac{\partial g}{\partial w} = x$$ (w는 1이되고, x는 상수이니 $$1*x = x$$)
* x를 기준으로 g를 편미분하면 $$ \frac{\partial g}{\partial x} = w  $$ 

###### 위 치환 식 $$f(x) = g + b$$ 에서 
* $$ \frac{\partial f}{\partial g} = 1$$
* $$ \frac{\partial f}{\partial b} = 1$$
