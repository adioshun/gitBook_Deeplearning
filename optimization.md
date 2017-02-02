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

> 


# Overfitting 문제 해결법 
1. More Training data
2. Reduce the number of feature
3. Regularization (규제화)

## 1. Regularizaion

### 1.1 Weight Decay(가중치 감쇠 )
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

### 1.2 가중치 상한 
* 가중치 값의 상한을 통해 가중치를 제약하는 방법
* 가중치 감쇠보다 나은 성능 보임 
* Dropout 과 같이 사용 가능 



### 1.3 Dropout
![](/assets/dropout.PNG)
* 네트워크의 일부만 사용하여서 학습[^4] 
* (조심) Training 시에만 dropout_rate를 `~0.9`미만으로 적용하고, Evaluation 할때는 dropout_rate를 `1`로 적용

> 신경망의 일부를 학습 시에 랜덤으로 무효화 하는 유사 방법(트롭커넥트, 확률적 최대 풀링)들이 존재 하나, 사용 편의와 적용 범위로 볼때 DropOut이 효과적

#Ensemble 
* 여러 학습 모델을 생성하고 마지막에 합쳐서 결과를 산출
* 2~4,5%까지 성능 향상 가능
* 충분한 컴퓨팅 파워 필요 


# 미니배치(Minibatch)
* 샘플한개 단위가 아니라 몇 개의 샘플을 하나의 작은 집합으로 묶은 집합 단위로 가중치를 업데이트 한다. 
* 복수의 샘플을 묶은 작은 집합을 `미니배치`라고 부른다. 


---




[^1]: Hinton et al.,"A Fast Learning Algorithm for Deep Belief Nets", 2006
[^2]: X.Glorot and Y.Bengio, "understanding the difficulty of training deep feedforward neural networks", 2010
[^3]: K.He, "Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification", 2015
[^4]: Srivastava et al., "A Simple way to Prevent Neural Networks from Overfitting", 2014