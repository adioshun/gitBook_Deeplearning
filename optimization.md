# [참고] 

## Weight Initial 하기 
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


---




[^1]: Hinton et al.,"A Fast Learning Algorithm for Deep Belief Nets", 2006
[^2]: X.Glorot and Y.Bengio, "understanding the difficulty of training deep feedforward neural networks", 2010
[^3]: K.He, "Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification", 2015