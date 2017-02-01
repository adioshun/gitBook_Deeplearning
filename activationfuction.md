# Sigmoid 함수





# ReLu(Rectified Linear Unit)

![Vanishing gradien](/assets/vgpro.PNG)

* Vanishing gradient문제 : Deep-wide 네트워크에서 Back propagation할경우 sigmoid의 0~1의 문제로 제대로 Input Layer까지 전달 안됨 [[Youtube 추가설명]](https://youtu.be/cKtg_fpw88c?t=7m9s)


* Vanishing gradient 해결책 : 0보다 작으면 off, 0보다 크면 계속 증가 (ReLu)



> ReLu를 사용하더라라도 구현시 마지막 레이어에서는 0~1사이의 값이어야 하므로 Sigmoid 사용


# Leaky ReLu
* O보다 작으면 Off하지 말고, 작은 폭의 -를 가지게 하자  
* Max(0.1x, x)

# 그외 Activation Function 
![](/assets/acode.PNG)




---
# [참고] Vanishing gradient 발생 2번째 이유 
* Initial weight를 초기값을 램덤하게 설정

## Weight Initial 방법 
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