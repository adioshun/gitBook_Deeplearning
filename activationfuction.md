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
