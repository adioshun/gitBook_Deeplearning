# Weight Initial 하기 
> Hilton 교수의 3번째 제안
```
기존 : Initial weight를 초기값을 램덤하게 설정
* Deep-wide Network가 정확도가 안 맞는 2번째 이유 
```
1. 0이 아닌 값으로 선정 
2. RBM(restricted boltzmann machine﻿)을 이용[^1] 
    * 근접 레이어간 Pre-training(Forward/Backward)를 하면서 결과값을 비교 하면서 wight를 수정 
    * 이렇게 생성된 네트워크를 `Deep Belief Network`라 부름 
    * 연산이 오래 걸리고, 다른 좋은 방법들이 나와서 요즘 사용 안함 -> Xavier Initialization/He's Initialization 
    * 이러한 연산을 오토 인코더/오토 디코더라고도 함
3. Xavier Initialization/He's Initialization :입력과 아웃의 갯수를 사용하여 결정[^2],[^3]
    * W 정의시 input(fan_in)으로 Output(fan_out)으로 정의 하는것만으로도 RBM과 같은 성능 보임[[Youtbue설명](https://youtu.be/4rC0sWrp3Uw?t=10m42s)
    * Xavier : `random(fan_in, fan_out)/np.sqrt(fan_in)`
    * He : `random(fan_in, fan_out)/np.sqrt(fan_in/2)`
4. Batch normalization 
5. layer sequential Uniform variance 

> Weight Initial는 아직 Active research area임 

# 연구 결과들 (초기값 설정)
![](/assets/Screenshot from 2017-02-21 05-32-14.png)