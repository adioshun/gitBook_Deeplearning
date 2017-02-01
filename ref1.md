## 정의 
* 미분 : 미세한게 나누다 
![](/assets/def23.PNG)
* 변화량과 변화량을 나누다  $$ \frac{\Delta y}{\Delta x} $$
* 미세하게 나눈다 = $$ \Delta x \rightarrow 0$$으로 보낸다 =x의 변화량을 0처럼 미세하게 한계치($$\lim_{\Delta x \rightarrow 0}$$)까지로 만든다.  

$$
\lim_{h \rightarrow 0}\frac{\Delta y}{\Delta x} = 변화량과 변화량을 미세하게 나누어서 값을 구한다. 
$$

* 결국, 접선의 기울기(=순간변화율=미분 계수) 
 * x에 대한 기울기값f(x) 출력 - 함수역할 - __도함수__라 부름 
 * cf. 평균변화율 = 직선의 기울기

## 공식 유도 
1. $$ 원 공식 \frac{f(b)-f(a)}{b-a} 에서 $$

2. $$ b는 a에서 \Delta x만큼 이동한 것이므로 b= a+\Delta x로 치환  $$ 

3. $$ 즉, \frac{f(a+\Delta x)-f(a)}{a+\Delta x-a} = \frac{f(a+\Delta x)-f(a)}{\Delta x} $$

4. 표현의 간결성을 위해 $$ \Delta x$$를 $$h$$로 치환 = $$\frac{f(a+h)-f(a)}{h} $$


    
```
 미분이라는 행위를 했을 때 나온 식이 도함수고 
 이를 함수라 표현한 이유가 어떤 값하나에 다른 하나의 식이 나오니까 도함수(x를 넣으면 그때의 미분계수/기울기를 알수 있는)라 칭함
```
 
### 표기법 1
* $$ \Delta x$$는 x의 변화, 미세한 x의 변화량 $$dx$$ 로 표시(d=differentiation)
* $$ \frac{dy}{dx} $$ = y라는 함수를 x를 변수로 해서 미분 하시오 

### 표기법 2
$$ f^\prime (a) $$ = a에서 f의 미분값


### 표기법 3 

![](/assets/def231.PNG)
1. $$\Delta x$$를 $$h$$로 치환 
2. $$x$$의 변화량 $$h$$, 이때 $$y$$의 변화량은 $$ f(a+h) $$

$$
즉, \frac{dy}{dx} = f^\prime (a) = \lim_{h \rightarrow 0} \frac{f(a+h)-f(a)}{h} 

$$



## 미분을 나타내는 방법

$$
 \lim_{h \rightarrow 0}\frac{A-B}{C} : A에서 B를 뺀게 C에 있게되면 미분을 나타내는 것임 
$$ 


###### 식 1
$$
\lim_{h \rightarrow 0}\frac{f(x+h)-f(x)}{h} = \frac{f(x)-f(x+h)}{-h} 
$$

##### 식2
$$
\lim_{h \rightarrow a}\frac{f(x)-f(a)}{x-a} = \frac{f(a)-f(x)}{a-x}
$$

## 미분과 기울기
![](/assets/decens.PNG)

기울기 $$= \frac{높이}{밑변의 변화량} = \frac{f(b)-f(a)}{b-a} = \frac{\Delta y}{\Delta x}$$

```
b-a : x의 증가량
f(b)-f(a) : y의 증가량 
```

식 1과는 일치 

식 2도 풀어 쓰면 아래와 같으므로 일치 
$$
\lim_{h \rightarrow 0}\frac{f(x+h)-f(x)}{h} = \frac{f(x)-f(x+h)}{-h} = \frac{f(x+h)-f(x)}{(x+h)-(x)} 
$$


---
> 참고
> * [모두를 위한 딥러닝 Lec9x: 10분안에 미분 정리하기](https://youtu.be/oZyvmtqLmLo?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm)
> * [꼼수학 : 미분의 기초](https://youtu.be/xXvnfqr5b3A)
> * [장자윤 : 미분의 정의와 뜻](https://www.youtube.com/watch?v=cr_SVH27n4c)