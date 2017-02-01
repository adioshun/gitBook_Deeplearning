# Weight Initial 하기 
```
기존 : Initial weight를 초기값을 램덤하게 설정
* Deep-wide Network가 정확도가 안 맞는 2번째 이유 
```
1. 0이 아닌 값으로 선정 
2. RBM(restricted boltzmann machine﻿)을 이용[^1] 
    * 근접 레이어간 Pre-training(Forward/Backward)를 하면서 결과값을 비교 하면서 wight를 수정 
    * 이렇게 생성된 네트워크를 `Deep Belief Network`라 부름 
    * 연산이 오래 걸리고, 다른 좋은 방법들이 나와서 요즘 사용 안함
3. Xavier Initialization/He's Initialization :입력과 아웃의 갯수를 사용하여 결정[^2],[^3]
    * Xavier : `random(fan_in, fan_out)/np.sqrt(fan_in)`
    * He : `random(fan_in, fan_out)/np.sqrt(fan_in/2)`
4. Batch normalization 
5. layer sequential Uniform variance 

> Weight Initial는 아직 Active research area임 



# Dropout
## 1. 기존 Overfitting 문제 해결법 
1. More Training data
2. Reduce the number of feature
3. Regularization 
    * weight에 너무 큰값 주지 않기 (큰값 = 구부려짐 커짐)
    * L2Regulization = $$ cost + \rightthreetimes \sum w^2 $$
    * TF코드 `l2reg=0.001*tf.reduce_sum(tf.square(w))`
    * NN에서 Regularization 방법중 `__Dropout__`이 있음
 

## 2. Regularizaion : Dropout
![](/assets/dropout.PNG)
* 네트워크의 일부만 사용하여서 학습[^4] 
* (조심) Training 시에만 dropout_rate를 `~0.9`미만으로 적용하고, Evaluation 할때는 dropout_rate를 `1`로 적용

#Ensemble 
* 여러 학습 모델을 생성하고 마지막에 합쳐서 결과를 산출
* 2~4,5%까지 성능 향상 가능
* 충분한 컴퓨팅 파워 필요 




---




[^1]: Hinton et al.,"A Fast Learning Algorithm for Deep Belief Nets", 2006
[^2]: X.Glorot and Y.Bengio, "understanding the difficulty of training deep feedforward neural networks", 2010
[^3]: K.He, "Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification", 2015
[^4]: Srivastava et al., "A Simple way to Prevent Neural Networks from Overfitting", 2014