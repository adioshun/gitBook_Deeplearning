# Autoencoder

Autoencoder는 기존의 Neural Network의 **Unsupervised learning** 버젼이다. 

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


## 1. 목적

출력값 $$ \hat{x} $$를 입력값 $$ x $$와 유사하게 만들고자 하는걸 목표로 함
  * 학습의 목표는 입력을 부호화한 뒤, 이어 다시 복호화 했을 때 원래의 입력을 되도록 충실히 재현할 수 있는 부호화 방법을 찾는것이다. 

##### 목적1 : 특징 표현력을 가진 신경망

- 손글씨등 개인 차이 혹은 노이즈가 포함된 상태에서도 바르게 인식 할수 있는 특징으로 변환 할수 있음 
  - 압축/복원이 가능할 정도로 특징 표현력을 가진 신경망 구축 
  - 노이즈 제거 

##### 목적2 : 초기 파라미터 값 획득

- 딥 네트워크의 사전훈련, 즉 그 가중치의 좋은 초기값을 얻는 목적으로 사용된다.


## 2. 구조

* 제약 볼츠만 머신과 같은 2개 층(입력층 + 출력층) 


### 2.1 오토인코더 설계

#### A. 출력층의 활성화 함수

* 중간층의 활성화 함수\($$ f $$\) : 자유롭게 변경 가능, 통상적으로 비선형함수

* 출력층의 활성화 함수\($$ \tilde{f}$$\) : 입력 데이터의 유형에 따라 선택 \(신경망의 목표 출력이 입력한 자신이 될수 있도록\)
  * 실수값 : 항등사항
  * 이진값 : 로지스틱

#### B. 출력층의 오차 함수

* 출력층의 활성화 함수에 따라 오차 함수 결정 
  * 실수값 : 입력/출력값의 차에 대한 제곱합
  * 이진값 : 교차 엔트로피 

## 3. 종류

```
1  Auto-Encoder (Basic form)
2. Stacked Auto-Encoder : 적층 자기 부호화기, 초기 파라미터 획 
3. Sparse Auto-Encoder : 희소 자기 부호화기 
4. Denoising Auto-Encoder (dA) : 노이즈 제거 
5. Stacked Denoising Auto-Encoder (SdA)
```

### 3.1  Auto-Encoder \(Basic form\)

![오토인코더 Basic Form](https://wikidocs.net/images/page/3413/AE.png)

* 초창기에 기본적인 Auto-Encoder 의 Weight를 학습하는 방법
  1. BP\(Backpropagation\) 알고리즘
  2. SGD\(Stochastic Gradient Descent\)알고리즘 

### 3.2 Stacked Auto-Encoder

![](https://wikidocs.net/images/page/3413/stackedAE.png)

* Stacked Autoencoder가 Autoencoder에 비해 갖는 가장 큰 차이점은 DBN\(Deep Belief Network\) \[Hinton 06\] 의 구조라는 것이다.
* AE를 여러개 쌓아 놓은 형태가 된다. 가장 압축된 레이어를 Bottleneck hidden layer라고 한다. 
* 학습 방법 : [Greedy layer-wise Training](http://m.blog.naver.com/laonple/220884698923#)

### 3.3 Sparse Auto-Encoder

![](https://wikidocs.net/images/page/3413/sparseAE.png)

> 추가 설명 : 라온피플 블로그 [Sparse coding](http://m.blog.naver.com/laonple/220914873095), [Sparse Autoencoding](http://m.blog.naver.com/laonple/220943887634)  
> 딥러닝 제대로 시작하기 80page

### 3.4 Denoising Auto-Encoder \(dA\)

![](https://wikidocs.net/images/page/3413/denoisingAE.png)

Denoising Auto-Encoder는 데이터에 Noise 가 추가되었을 때, 이러한 Noise를 제거하여 원래의 데이터를 Extraction하는 모델이다.  
실제 사물 인식이라든지, Vision을 연구하는 분야에서는 고의적으로 input data에 noise를 추가하고, 추가된 노이즈를 토대로 학습된 데이터에서 나오는 결과값이, 내가 노이즈를 삽입하기 전의 pure input 값인지를 확인한다.

> 추가 설명 : [라온피플 블로그](http://m.blog.naver.com/laonple/220891144201)

### 3.5 Stacked Denoising Auto-Encoder \(SdA\)

![](https://wikidocs.net/images/page/3413/sDA.png)

---

[솔라리스의 인공지는 연구실](http://solarisailab.com/archives/113): AutoEncoders & Sparsity  
[위키독스의 Introduction Auto-Encoder](https://wikidocs.net/3413)  
[Autoencoder vs RBM \(+ vs CNN\)](http://khanrc.tistory.com/entry/Autoencoder-vs-RBM-vs-CNN)  
[번역: A Deep Learning Tutorial: From Perceptrons to Deep Networks](http://khanrc.tistory.com/entry/Deep-Learning-Tutorial)  
[라오피플 블로그 : AutoEnoder 1~5](http://m.blog.naver.com/laonple/220880813236)  
[Keras를 이용한 Autoencoder구현 코드](https://byeongkijeong.github.io/Keras-Autoencoder/)

---

```python
# -*- coding: utf-8 -*-
# 대표적인 비감독(Unsupervised) 학습 방법인 Autoencoder 를 사용해봅니다.

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


#########
# 옵션 설정
######
learning_rate = 0.01
training_epoch = 20
batch_size = 100
# 신경망 레이어 구성 옵션
n_hidden = 256  # 히든 레이어의 특성 갯수
n_input = 28*28   # 입력값 크기 - 이미지 픽셀수


#########
# 신경망 모델 구성
######
# Y 가 없습니다. 입력값을 Y로 사용하기 때문입니다.
X = tf.placeholder(tf.float32, [None, n_input])

# 인코더 레이어와 디코더 레이어의 가중치와 편향 변수를 설정합니다.
# 다음과 같이 이어지는 레이어를 구성하기 위한 값들 입니다.
# input -> encode -> decode -> output
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
# encode 의 아웃풋 크기를 입력값보다 작은 크기로 만들어 정보를 압축하여 특성을 뽑아내고,
# decode 의 출력을 입력값과 동일한 크기를 갖도록하여 입력과 똑같은 아웃풋을 만들어 내도록 합니다.
# 히든 레이어의 구성과 특성치을 뽑아내는 알고리즘을 변경하여 다양한 오토인코더를 만들 수 있습니다.
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))

# sigmoid 함수를 이용해 신경망 레이어를 구성합니다.
# sigmoid(X * W + b)
# 인코더 레이어 구성
encoder = tf.nn.sigmoid(
                tf.add(tf.matmul(X, W_encode), b_encode))
# 디코더 레이어 구성
# 이 디코더가 최종 모델이 됩니다.
decoder = tf.nn.sigmoid(
                tf.add(tf.matmul(encoder, W_decode), b_decode))

# 디코더는 인풋과 최대한 같은 결과를 내야 하므로,
# 디코딩한 결과를 평가하기 위해 (손실 함수 구성을 위해)
# 입력 값인 X 값을 평가를 위한 실측 결과 값으로 설정합니다. (안해도 됩니다만, 이해를 위해 작성)
Y = X

cost = tf.reduce_mean(tf.pow(Y - decoder, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(training_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        total_cost += cost_val

    print 'Epoch:', '%04d' % (epoch + 1), \
        'Avg. cost =', '{:.6f}'.format(total_cost / total_batch)

print '최적화 완료!'


#########
# 결과 확인
# 입력값(위쪽)과 모델이 생성한 값(아래쪽)을 시각적으로 비교해봅니다.
######
sample_size = 10

samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

plt.show()
```



