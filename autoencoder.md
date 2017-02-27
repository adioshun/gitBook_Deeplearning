# Autoencoder

딥러닝은 크게 아래 두 Phase로 나눌수 있다. 

1. unsupervised pretraining 페이즈 : Input을 reconstruction 할 수 있는 hidden units를 찾아내는 과정
    * 대표적 모델 : 오토 인코더 & RBM
2. supervised fine-tuning 페이즈 : SVM이나 softmax 등의 classifier를 학습

```
[RBM]
- Generative model(에너지 모델을 차용하고 그 에너지 모델은 볼츠만 분포에 기반)
- 확률분포에 기반하여 visible 변수들과 hidden 변수들간에 어떤 상관관계가 있고 어떻게 상호작용하는지를 파악하는 개념
 * 인풋 (visible) 변수와 히든 변수의 joint probability distribution을 학습하고자 하는 것
- 오토인코더와는 달리 찾아낸 확률분포로부터 새로운 데이터를 생성할 수 있다. 파라메터가 많은만큼 더욱 유연

[오토 인코더]
- Deterministic Model
- 피처를 축소하는 심플한 개념
- 오토인코더가 직관적일 뿐 아니라 구현하기도 더 쉽고 파라메터가 더 적어서 튜닝하기가 쉽다
```

> Unsupervised Pretraining : 2006년 이후로 심층네트워크 대한 학습방법의 정석으로 인정 되었지만, 2010년이 되면서 ReLu, Dropout, Maxout, Batch Normalization등의 방법을 이용하여도 성능이 좋아 사용 하지 않음 (굳이 PreTraining을 하지 않아도 Supervised방식으로도 좋은 성과 보임)

## 1. 목적 
* autoencoder는 기존의 Neural Network의 Unsupervised learning 버젼이다. 
* 오토인코더는 전형적인 FNN인데, 데이터셋을 압축적이고 분배된 표현(인코딩) 으로 학습하는 것을 목표
* 기존에 대부분 데이터의 압축을 위해 활용되었으나, 최근에는 딥 러닝 (deep learning)에 대한 연구가 활발해지면서 입력 벡터의 차원 축소, 은닉층의 학습 등에 많이 이용
* 출력값 $$ \hat{x} $$를 입력값 $$ x $$와 유사하게 만들고자 하는걸 목표로 함
    * 학습의 목표는 입력을 부호화한 뒤, 이어 다시 복호화 했을 때 원래의 입력을 되도록 충실히 재현할 수 있는 부호화 방법을 찾는것이다. 

> 자기부호화기(오토인코더)는 목표 출력 없이 입력만으로 구성된 훈련 데이터로 비지도학습을 수행하여 데이터의 특징을 잘 나타내는, 더 나은 표현을 얻는것이 목표인 신경망이다. 
> * 딥 네트워크의 사전훈련, 즉 그 가중치의 좋은 초기값을 얻는 목적으로 사용된다. 

### 1.1 데이터를 나타내는 특징을 학습 
* 샘플 x의 또 다른 표현인 y를 얻는것 

### 1.2 주성분 분석과의 관계
* 나중에 다시 살펴 보기 

### 1.3 


## 2. 구조 
* 일반적인 FNN과 비슷 
* 단, 입력층과 츨력층의 크기가 항상 같다[1]

![](http://cfile9.uf.tistory.com/image/266B1740579DA3B3080567)

### 2.1 오토인코더 설계

#### A. 출력층의 활성화 함수
* 중간층의 활성화 함수($$ f $$) : 자유롭게 변경 가능, 통상적으로 비선형함수

* 출력층의 활성화 함수($$ \tilde{f}$$) : 입력 데이터의 유형에 따라 선택 (신경망의 목표 출력이 입력한 자신이 될수 있도록)
    * 실수값 : 항등사항
    * 이진값 : 로지스틱


#### B. 출력층의 오차 함수
* 출력층의 활성화 함수에 따라 오차 함수 결정 
    * 실수값 : 입력/출력값의 차에 대한 제곱합
    * 이진값 : 교차 엔트로피 
    
## 3. 종류 
    1  Auto-Encoder (Basic form)
    2. Stacked Auto-Encoder : Hidden 레이어를 여러개 쌓아서 구현 
    3. Sparse Auto-Encoder
    4. Denoising Auto-Encoder (dA)
    5. Stacked Denoising Auto-Encoder (SdA)


### 3.1  Auto-Encoder (Basic form)

![오토인코더 Basic Form](https://wikidocs.net/images/page/3413/AE.png)

* 초창기에 기본적인 Auto-Encoder 의 Weight를 학습하는 방법
    1. BP(Backpropagation) 알고리즘
    2. SGD(Stochastic Gradient Descent)알고리즘 


### 3.2 Stacked Auto-Encoder
![](https://wikidocs.net/images/page/3413/stackedAE.png)
* Stacked Autoencoder가 Autoencoder에 비해 갖는 가장 큰 차이점은 DBN(Deep Belief Network) [Hinton 06] 의 구조라는 것이다.
* AE를 여러개 쌓아 놓은 형태가 된다. 가장 압축된 레이어를 Bottleneck hidden layer라고 한다. 
* 학습 방법 : [Greedy layer-wise Training](http://m.blog.naver.com/laonple/220884698923#)

### 3.3 Sparse Auto-Encoder
![](https://wikidocs.net/images/page/3413/sparseAE.png)

* Dropout이 이건가? 


### 3.4 Denoising Auto-Encoder (dA)
![](https://wikidocs.net/images/page/3413/denoisingAE.png)

Denoising Auto-Encoder는 데이터에 Noise 가 추가되었을 때, 이러한 Noise를 제거하여 원래의 데이터를 Extraction하는 모델이다.
실제 사물 인식이라든지, Vision을 연구하는 분야에서는 고의적으로 input data에 noise를 추가하고, 추가된 노이즈를 토대로 학습된 데이터에서 나오는 결과값이, 내가 노이즈를 삽입하기 전의 pure input 값인지를 확인한다.

### 3.5 Stacked Denoising Auto-Encoder (SdA)
![](https://wikidocs.net/images/page/3413/sDA.png)

---
[1]: http://untitledtblog.tistory.com/92 "[머신러닝] - Autoencoder" 

[솔라리스의 인공지는 연구실](http://solarisailab.com/archives/113): AutoEncoders & Sparsity
[위키독스의 Introduction Auto-Encoder](https://wikidocs.net/3413)
[Autoencoder vs RBM (+ vs CNN)](http://khanrc.tistory.com/entry/Autoencoder-vs-RBM-vs-CNN)
[번역: A Deep Learning Tutorial: From Perceptrons to Deep Networks](http://khanrc.tistory.com/entry/Deep-Learning-Tutorial)
[라오피플 블로그 : AutoEnoder 1~5](http://m.blog.naver.com/laonple/220880813236)