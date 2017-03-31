# Overfitting 문제 해결법 
1. More Training data
2. Reduce the number of feature
3. __Regularization (규제화)__ : 학습시 가중치의 자유도 제약 
    * Weight
    * Dropout 
    


## 1. 규제화 방법 #1 : Weight Decay(가중치 감쇠)
* 가중치에 어떤 제약을 가하는것
    * 학습시 w에 대한 자유도 제약
    * weight에 너무 큰값 주지 않기 (큰값 = 구부려짐 커짐)
* 오차함수에 가중치의 제곱합(norm의 제곱)을 더한 뒤, 이를 최소화
* $$ \rightthreetimes$$는 규제화의 강도를 제어하는 파라미터 
    * $$ \rightthreetimes$$ = 0.01 ~0.00001 범위에서 선택 
    * L2Regulization = $$ cost + \rightthreetimes \sum w^2 $$
* 결과적으로 가중치는 자신의 크기에 비례하는 속도롤 항상 감쇠 
* Weight Decay는 신경망의 가중치w에만 작용하며, 바이어스b에는 적용하지 않는다. 


```
TF코드 `l2reg=0.001*tf.reduce_sum(tf.square(w))`
```

## 2. 규제화 방법 #2 : 가중치 상한 
* 가중치 값의 상한을 통해 가중치를 제약하는 방법
* 가중치 감쇠보다 나은 성능 보임 
* Dropout 과 같이 사용 가능 



## 3. 규제화 방법 #3 : Dropout 
![](/assets/dropout.PNG)
* 네트워크의 일부만 사용하여서 학습[1] 
* (조심) Training 시에만 dropout_rate를 `~0.9`미만으로 적용하고, Evaluation 할때는 dropout_rate를 `1`로 적용

> 신경망의 일부를 학습 시에 랜덤으로 무효화 하는 유사 방법(트롭커넥트, 확률적 최대 풀링)들이 존재 하나, 사용 편의와 적용 범위로 볼때 DropOut이 효과적



---





[1]: Srivastava et al., "A Simple way to Prevent Neural Networks from Overfitting", 2014

--- 
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)