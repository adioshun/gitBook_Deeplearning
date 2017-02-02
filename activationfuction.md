# Activation Function
일반적으로 단조증가(Monotone increasing)하는 비선형함수가 사용됨다. 



## 1. Sigmoid 함수
* Logistic sigmoid function = Logistic function 

* Vanishing gradient문제 발생: Deep-wide 네트워크에서 Back propagation할경우 sigmoid의 0~1의 문제로 제대로 Input Layer까지 전달 안됨 [[Youtube 추가설명]](https://youtu.be/cKtg_fpw88c?t=7m9s)
* 너무 커질경우 0이나 1이 되어 버림 

![Vanishing gradien](/assets/vgpro.PNG)



> sigmoid 함수를 개선한 tanh 함수 사용 



## 2. ReLu(Rectified Linear Unit)
* 램프(ramp)함수를 이용 
* 0보다 작으면 off, 0보다 크면 계속 증가 (ReLu)
* 단순하지만, 계산량이 적으며 학습이 빠르고 결과도 좋음 



> ReLu를 사용하더라라도 구현시 마지막 레이어에서는 0~1사이의 값이어야 하므로 Sigmoid 사용


## 3. Leaky ReLu
* O보다 작으면 Off하지 말고, 작은 폭의 -를 가지게 하자  
* Max(0.1x, x)


## 5. 항등사상(Identity mapping)
* 회귀문제를 위한 신경망에서는 출력층으로 항등사상 사용


## 6. 소프트 맥스
* 클래스 분류를 목적으로 하는 신경망에서는 출력층으로 소프트맥스 함수 사용 

## 9. 그외 Activation Function 
![](/assets/acode.PNG)




---
