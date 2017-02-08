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
* 소프트맥스는 입력값을 지수화한 뒤 정규화 하는 과정 [[설명]](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/beginners/)
    * 지수화란 증거값을 하나 더 추가하면 어떤 가설에 대해 주어진 가중치를 곱으로 증가시키는 것을 의미합니다. 또한 반대로, 증거값의 갯수가 하나 줄어든다는 것은 가설의 가중치가 기존 가중치의 분수비로 줄어들게 된다는 뜻입니다. 
    * 어떤 가설도 0 또는 음의 가중치를 가질 수 없습니다. 
    * 그런 뒤 소프트맥스는 가중치를 정규화한 후, 모두 합하면 1이 되는 확률 분포로 만듭니다 [참고](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)


## 9. 그외 Activation Function 
![](/assets/acode.PNG)




---
