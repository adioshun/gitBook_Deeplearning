## 정의 
* 미분 : 미세한게 나누다 
![](/assets/def23.PNG)
* 변화량과 변화량을 나누다  $$ \frac{\Delta y}{\Delta x} $$
* 미세하게 나눈다 = $$ \Delta x \rightarrow 0$$으로 보낸다 =x의 변화량을 0처럼 미세하게 한계치($$\lim_{\Delta x \rightarrow 0}$$)까지로 만든다.  

$$
\lim_{h \rightarrow 0}\frac{\Delta y}{\Delta x} = 변화량과 변화량을 미세하게 나누어서 값을 구한다. 
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

기울기 $$= \frac{높이}{밑변의 변화량} = \frac{f(b)-f(a)}{b-a} $$

식 1과는 일치 

식 2도 풀어 쓰면 아래와 같으므로 일치 
$$
\lim_{h \rightarrow 0}\frac{f(x+h)-f(x)}{h} = \frac{f(x)-f(x+h)}{-h} = \frac{f(x+h)-f(x)}{(x+h)-(x)} 
$$


---
> 참고
> * [모두를 위한 딥러닝 Lec9x: 10분안에 미분 정리하기](https://youtu.be/oZyvmtqLmLo?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm)
> * [꼼수학 : 미분의 기초](https://youtu.be/xXvnfqr5b3A)