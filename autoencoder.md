# Autoencoder



* autoencoder는 기존의 Neural Network의 Unsupervised learning 버젼이다. 
* 출력값 $$ \hat{x} $$를 입력값 $$ x $$와 유사하게 만들고자 하는걸 목표로 함

* 종류 
    1  Auto-Encoder (Basic form)
    2. Stacked Auto-Encoder
    3. Sparse Auto-Encoder
    4. Denoising Auto-Encoder (dA)
    5. Stacked Denoising Auto-Encoder (SdA)


## 1  Auto-Encoder (Basic form)

![오토인코더 Basic Form](https://wikidocs.net/images/page/3413/AE.png)

* 초창기에 기본적인 Auto-Encoder 의 Weight를 학습하는 방법
    1. BP(Backpropagation) 알고리즘
    2. SGD(Stochastic Gradient Descent)알고리즘 


## 2. Stacked Auto-Encoder
![](https://wikidocs.net/images/page/3413/stackedAE.png)
* Stacked Autoencoder가 Autoencoder에 비해 갖는 가장 큰 차이점은 DBN(Deep Belief Network) [Hinton 06] 의 구조라는 것이다.


## 3. Sparse Auto-Encoder
![](https://wikidocs.net/images/page/3413/sparseAE.png)

* Dropout이 이건가? 


## 4. Denoising Auto-Encoder (dA)
![](https://wikidocs.net/images/page/3413/denoisingAE.png)

Denoising Auto-Encoder는 데이터에 Noise 가 추가되었을 때, 이러한 Noise를 제거하여 원래의 데이터를 Extraction하는 모델이다.
실제 사물 인식이라든지, Vision을 연구하는 분야에서는 고의적으로 input data에 noise를 추가하고, 추가된 노이즈를 토대로 학습된 데이터에서 나오는 결과값이, 내가 노이즈를 삽입하기 전의 pure input 값인지를 확인한다.

## 5. Stacked Denoising Auto-Encoder (SdA)
![](https://wikidocs.net/images/page/3413/sDA.png)















---
[솔라리스의 인공지는 연구실](http://solarisailab.com/archives/113): AutoEncoders & Sparsity
[위키독스의 Introduction Auto-Encoder](https://wikidocs.net/3413)