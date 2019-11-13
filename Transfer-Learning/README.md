# Transfer Learning

> 참고 코드 : https://jeinalog.tistory.com/entry/Transfer-Learning-%ED%95%99%EC%8A%B5%EB%90%9C-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EC%83%88%EB%A1%9C%EC%9A%B4-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%EC%97%90-%EC%A0%81%EC%9A%A9%ED%95%98%EA%B8%B0 


## 1. 개요 

### 1.1 역사 
- 1995년 NIPS workshop에서 "Learning to learn"이라는 주제로 소개
- Domain Adaptation : knowledge transfer
- Transfer learning

### 1.2 장점

높은 정확도를 비교적 짧은 시간 내에 달성 

### 1.3 방법

사전학습 된 모델 (pre-trained model) 을 이용, eg. 밑바닥에서부터 모델을 쌓아올리는 대신에 이미 학습되어있는 패턴들을 활용 

### 1.4 제약

해결하려는 문제와 비슷한 유형에 적용 

> 만약 이미지셋이 100만개보다 적다면 pre-train 모델을 사용하라!!!


### 1.5 [pipeline](https://jeinalog.tistory.com/entry/Transfer-Learning-%ED%95%99%EC%8A%B5%EB%90%9C-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EC%83%88%EB%A1%9C%EC%9A%B4-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%EC%97%90-%EC%A0%81%EC%9A%A9%ED%95%98%EA%B8%B0)
1. 원래 모델에 있던 classifier를 없애는 것
2. 내 목적에 맞는 새로운 classifier를 추가
3. 다음 세 가지 전략 중 한 가지 방법을 이용해 파인튜닝(fine-tune)을 진행
    - 전략 1 : 전체 모델을 새로 학습시키기
    - 전략 2 : Convolutional base의 일부분은 고정시킨 상태로, 나머지 계층과 classifier를 새로 학습시키기
    - 전략 3 : Convloutional base는 고정시키고, classifier만 새로 학습시키기
    
    
![](https://i.imgur.com/gQfiI5J.png)

### 1.6 용어 

- Finetuning : 모델의 파라미터를 미세하게 조정하는 행위, 이미 존재하는 모델에 추가 데이터를 투입하여 파라미터를 업데이트하는 것을 말한다

- 보틀넥 피쳐(Bottleneck feature) : 모델의 마지막 부분으로 보통 Pooling 다음 FC 전, 모델에서 가장 추상화된 피쳐

## 2. 적용 방법(상세) 

![](https://i.imgur.com/zHBvb8x.png)

### 2.1 사전 학습 모델 선별 

다양하게 공개되어 있는 사전학습 모델 중에서, 내 문제를 푸는 것에 적합해보이는 모델을 선택

### 2.2 적용 시나리오 선별 하기 

> 내 문제가 [데이터크기(약, 1000개)-유사성] 그래프에서 어떤 부분에 속하는지 알아보기


## 2.3 내 모델을 Fine-tuning 하기

##### Quadrant 1 : 크기가 크고 유사성이 작은 데이터셋일 때 -> 전략 1 
많은 레이어를 fine-tune 한다.

- 충분한 데이터가 있으므로 그냥 처음부터 학습 하는게 좋을수 있다.
- 그러나 현실에서는 전체 층에 대해서 Fine-Tune을 수행 한다.


##### Quadrant 2 : 크기가 크고 유사성도 높은 데이터셋일 때 -> ALL (추천 : 전략 2)  

추가한 레이어와 몇개 레이어를 fine-tune 한다.

- 추가 데이터가 생긴것과 비슷
- 더 많은 자료를 가졌기 때문에, 전체 망을 통해 Fine-tune을 시도한다면 과적합 없는 더 신뢰를 가질 수 있다.
- More High layer의 일부를 적절 제거 한후 새로 합습 한다. (FC 무조건 재 학습)

```python
for layer in model.layers:
   layer.trainable = True
#The default is already set to True. I have mentioned it here to make things clear.
for layer in model.layers[:5]:
   layer.trainable = False.
# Here I am freezing the first 5 layers 
```

> [코드 출처](https://medium.com/towards-data-science/transfer-learning-using-keras-d804b2e04ef8)


##### Quadrant 3 : 크기가 작고 유사성도 작은 데이터셋일 때 -> 전략 2 

큰일이다…..

- 데이터가 적기 때문에 only train a linear classifier하는것이 최선이다.
- 추천 : train the SVM classifier from activations somewhere earlier in the network
- 비추 : train the classifier form the top of the network(=more dataset-specific features)
- data augmentation도 고려 필요 




##### Quadrant 4 : 크기가 작지만 유사성은 높은 데이터셋일 때 -> 전략 3 

끝 레이어(top layer)에 도메인에 맞는 레이어를 추가하고 추가한 레이어만 학습한다.

- 데이터양이 적기 때문에 Fine Tune을 할경우 Overfitting될 우려가 있다. 
- 데이터가 기본 데이터셋과 비슷하므로 ConvNet의 Higher-Lever의 특징이 비슷하다고 가정 할수 있음 
- [결론] 새 classifier만 학습시키는 것


```

for layer in model.layers:
   layer.trainable = False
#Now we will be training only the classifiers (FC layers)

```
> [코드 출처](https://medium.com/towards-data-science/transfer-learning-using-keras-d804b2e04ef8)





```
1. [도메인이 기존 데이터셋과 매우 다르]XS≠XTXS≠XT. The feature spaces of the source and target domain are different, e.g. the documents are written in two different languages. In the context of natural language processing, this is generally referred to as cross-lingual adaptation.

2. P(XS)≠P(XT)P(XS)≠P(XT). The marginal probability distributions of source and target domain are different, e.g. the documents discuss different topics. This scenario is generally known as domain adaptation.

3. YS≠YTYS≠YT. The label spaces between the two tasks are different, e.g. documents need to be assigned different labels in the target task. In practice, this scenario usually occurs with scenario 4, as it is extremely rare for two different tasks to have different label spaces, but exactly the same conditional probability distributions.

4. P(YS|XS)≠P(YT|XT)P(YS|XS)≠P(YT|XT). The conditional probability distributions of the source and target tasks are different, e.g. source and target documents are unbalanced with regard to their classes. This scenario is quite common in practice and approaches such as over-sampling, under-sampling, or SMOTE are widely used.
```

> 출처 : http://sebastianruder.com/transfer-learning/index.html


> 출처 : http://nmhkahn.github.io/CNN-Practice

> 출처 : http://cs231n.github.io/transfer-learning








> 출처 : http://sebastianruder.com/transfer-learning/index.html

# Trasfer Learing 제약 
- 함부로 기존 네트워크를 변경 할수 없다. 하지만 일부 항목들은 가능한다(eg. 이미지 크기)
- Learging rate를 기본의 값보다 작게 가져라 [[출처]](http://ishuca.tistory.com/entry/CS231n-Transfer-Learning-and-Finetuning-Convolutional-Neural-Networks-%ED%95%9C%EA%B5%AD%EC%96%B4-%EB%B2%88%EC%97%AD)


#### Bottlenecking
- FC만 변경하여 학습 해도 시간이 많이 걸린다면 [Bottlenecking](https://medium.com/@galen.ballew/transferlearning-b65772083b47)기법 적용

- FC레이어가 없는 기존넷(VGG16)을 이용하여 `예측`한 값을 사용 
    - `bottleneck_features = model.predict_generator(generator,)
    - 학습보다 예측이 Back Propagation이 없으므로 속도가 빠름 


####  PathNet
인공지능은 전이학습을 통해 이전의 학습에서 배운 지식을 완전히 새로운 과제에 활용할 수 있음.

- 이전의 지식을 활용함으로써 인공지능은 새로운 과제에 새로이 신경망을 구축할 필요가 없으며, 전이학습은 과제 이행의 질과 시간적 효율성을 높여줌.

전이학습은 신경망을 구축하고 학습하는데 있어서 가장 큰 과제이며, DeepMind는 PathNet을 통해 이를 해결하기 위해 노력중임.
- PathNet이란 SGD(stochastic gradient descent)와 유전 선발방법(genetic selection method)을 사용하여 학습한 신경망 네트워크를 뜻함.
- PathNet은 여러 층(layer)의 모듈로 이루어져 있으며, 각각의 모듈은 다양한 형태의 신경망임(예: 나선형(convolutional), 피드포워드(feedforward) 또는 반복되는(recurrent) 형태).

학습을 하기에 앞서, 모듈에 대한 정의를 해야 함(L = 층수(number of layers), N = 하나의 층이 가질 수 있는 최대 모듈수).
- 마지막 층은 복잡하게 이루어져 있으며, 다른 과제와 함께 공유되지 않음.

- A3C(Advantage Actor-critic)를 활용하는 마지막 층은 가치함수(value function)와 정책평가(policy evaluation)을 나타냄.

- 모듈에 대한 정의가 이루어진 후, P 유전자형(또는 경로)가 네트워크에 생성됨. 각각의 유전자형 평가를 통해 최적화된 경로를 파악하고, 이 경로를 통해 학습을 힘.

- 경로의 학습은 백 프로퍼게이션(back propagation)을 사용한 SGD(stochastic gradient descent)를 통해 이루어지며, 한 번에 하나의 경로를 통해 실행됨.

과제를 학습한 후 네트워크는 최적 경로의 모든 매개변수(parameters)를 수정하며, 모든 매개변수는 다시 초기화 됨.

- A3C는 이전지식이 지워지지 않도록 이전과제의 최적경로가 수정되지 않도록 함.

> 출처 : [미래금융연구센터](http://www.kcft.or.kr/2017/02/2120)



---



# Advanced Transfer Learning(eg. Lifelong Learning)

## 정의 

Continual Learning(=Lifelong learning) : 옛 지식을 잊지 않으면서 새로운 지식을 학습하는 AI
- Incremental Training:새로운 데이터만을 사용, 기존 모델 재학습
    - 이전 데이터로부터 학습한 내용을 잊어버리는 현상인
    - Catastrophic Forgetting이 발생함

- Inclusive Training:전체 데이터를 사용하여 모델을 새롭게 학습
    - 전체 데이터에 대한 학습은 Scalability Issue가 있음

lifelong learning은 심층 신경망(DNN)에서 online/incremental learning의 특수한 사례로 생각할 수 있다.


Lifelong Machine Learning focuses on developing versatile systems that accumulate and refine their knowledge over time.

This research area integrates techniques from multiple subfields of Machine Learning and Artificial Intelligence, including 
- transfer learning, 
- multi-task learning, 
- online learning 
- and knowledge representation and maintenance.


## 용어 

- lifelong(=continual??) learning 

- transfer learning


-  sequential/Online/Incremental learning : model learns one observation at a time
	. sequential Vs. incremental =  데이터에 순서 법칙이 존재 할때 oder Vs. 데이터에 순서 법칙이 없을때 random 
	. online Vs.  incremental = Label정보 없음, 이전 내용을 잊을수 있음(Catastrophic Interference) Vs. 라벨 정보 있음, 이전 내용을 잊을 없음 

	. online Vs.  incremental = faster than the sampling rate VS. runs slower than the sampling rate(updating every 1000 samples)



https://datascience.stackexchange.com/questions/6186/is-there-a-difference-between-on-line-learning-incremental-learning-and-sequent

---

- [DEEP ONLINE LEARNING VIA META-LEARNING:CONTINUAL ADAPTATION FOR MODEL-BASED RL](https://arxiv.org/pdf/1812.07671.pdf)


- Sequential Labeling with online Deep Learning : [논문](https://arxiv.org/abs/1412.3397), [코드(Matlab)](https://github.com/ganggit/deepCRFs), 2014


- [어떻게 하면 싱싱한 데이터를 모형에 바로 적용할 수 있을까? – Bayesian Online Leaning](http://freesearch.pe.kr/archives/4497)


- [Object tracking by using online learning with deep neural network features](http://koasas.kaist.ac.kr/handle/10203/221670): 조영주, 2016


- [Online/Incremental Learning with Keras and Creme](https://www.pyimagesearch.com/2019/06/17/online-incremental-learning-with-keras-and-creme/): pyimagesearch, Creme는 머신러닝용인듯
- [Keras: Feature extraction on large datasets with Deep Learning](https://www.pyimagesearch.com/2019/05/27/keras-feature-extraction-on-large-datasets-with-deep-learning/)
- [Transfer Learning with Keras and Deep Learning](https://www.pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/)


- [OperationalAI: 지속적으로 학습하는 AnomalyDetection시스템 만들기](https://deview.kr/data/deview/2019/presentation/[143]DEVIEW2019_MakinaRocks_%E1%84%80%E1%85%B5%E1%86%B7%E1%84%80%E1%85%B5%E1%84%92%E1%85%A7%E1%86%AB.pdf) : DEVIEW2019, 김기현 

- [An Introduction to Online Machine Learning](https://medium.com/danny-butvinik/https-medium-com-dannybutvinik-online-machine-learning-842b1e999880) : blog 
- [Incremental Online Learning](https://medium.com/@dannybutvinik/incremental-online-learning-9868861db880):blog 


![](https://pbs.twimg.com/media/C7fbqxLW4AERcdD.jpg)

[Tensorflow 공식 Inception GitHUb](https://github.com/tensorflow/models/tree/master/inception)

# Article / post

- [A Gentle Intro to Transfer Learning](https://blog.slavv.com/a-gentle-intro-to-transfer-learning-2c0b674375a0)


- <del>[추천] CS231n : Transfer Learning and Fine-tuning Convolutional Neural Networks: [원문](http://cs231n.github.io/transfer-learning), [번역](http://ishuca.tistory.com/entry/CS231n-Transfer-Learning-and-Finetuning-Convolutional-Neural-Networks-한국어-번역)</del>

- <Del>[추천] [Transfer Learning - Machine Learning's Next Frontier](http://sebastianruder.com/transfer-learning/index.html)</del> 추천

- <del>[Using Transfer Learning and Bottlenecking to Capitalize on State of the Art DNNs](https://medium.com/@galen.ballew/transferlearning-b65772083b47), [Transfer Learning in Keras](https://galenballew.github.io//articles/transfer-learning/): 후반 텐서보드 코드  [GitHub](https://github.com/galenballew/transfer-learning)</del>

- <del>[TensorFlow 와 Inception-v3 를 이용하여 원하는 이미지 학습과 추론 해보기](http://gusrb.tistory.com/m/16): retrain.py코드를 이용한 실행, 코드 자체에 대한 이해필요</del>, [Google CodeLAb 실습](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html?index=..%2F..%2Findex#0)

- <del>[텐서플로우(TensorFlow)를 이용한 ImageNet 이미지 인식(추론) 프로그램 만들기](http://solarisailab.com/archives/346?ckattempt=1): solarisailab작성, 코드 포함</del>, 간단하나 TF코드 이해 필요

- <del>[CNNs in Practice](http://nmhkahn.github.io/CNN-Practice): 중간에 transfer-learning에 대한여 잠깐 언급 </del>

- <del>[DeepMind just published a mind blowing paper: PathNet.](http://www.kcft.or.kr/2017/02/2120): 미래금융연구센터 한글 정리 자료 </del>

- <del>[초짜 대학원생의 입장에서 이해하는 Domain-Adversarial Training of Neural Networks (DANN)](http://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural.html): 중간에 transfer-learning에 대한여 잠깐 언급</del>

- [PPT: Transfer Learning and Fine Tuning for Cross Domain Image Classification with Keras](https://www.slideshare.net/sujitpal/transfer-learning-and-fine-tuning-for-cross-domain-image-classification-with-keras)

- <del>[Transfer Learning using Keras](https://medium.com/towards-data-science/transfer-learning-using-keras-d804b2e04ef8)</del>

- [ImageNet: VGGNet, ResNet, Inception, and Xception with Keras](http://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)

- Transfer learning using pytorch: Vishnu Subramanian, [Part 1](https://medium.com/towards-data-science/transfer-learning-using-pytorch-4c3475f4495), [Part 2](https://medium.com/towards-data-science/transfer-learning-using-pytorch-part-2-9c5b18e15551)

- [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html): Keras 공식 자료, [[GitHub]](https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069), [[VGG16_weight]](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)

- [Transfer learning & The art of using Pre-trained Models in Deep Learning](https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/)

- [Globa Average Pooling](https://www.quora.com/What-is-global-average-pooling)을 사용함으로써 기존 학습시 사용했던 학습의 제약 없이 아무 Tensor/image size를 사용할수 있음

> What is important to know about global average pooling is that it allows the network to accept any Tensor/image size, instead of expecting the size that it was originally trained on.


# paper
- [A Survey on Transfer Learning ](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf):SJ Pan et al. 2010 [ppt](https://www.slideshare.net/azuring/a-survey-on-transfer-learning)

- [A survey of transfer learning](https://goo.gl/e87S9y): Karl Weiss, 2016, 40P

- [Transfer Schemes for Deep Learning in Image Classification, 2015,87P ](http://webia.lip6.fr/~carvalho/download/msc_micael_carvalho_2015.pdf):

- [How transferable are features in deep neural networks?,2014](https://arxiv.org/abs/1411.1792)

- [CNN Features off-the-shelf: an Astounding Baseline for Recognition,2014](https://arxiv.org/abs/1403.6382)

# Implementation

- [Udacity Deeplearning - Transfer Learning](https://github.com/udacity/deep-learning/tree/master/transfer-learning):

- [Convert Caffe models to TensorFlow.](https://github.com/ethereon/caffe-tensorflow)

- [Inception in TensorFlow](https://github.com/tensorflow/models/tree/master/inception): 텐서플로우 공식 코드

- [Jupyter: 08_Transfer_Learning](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/08_Transfer_Learning.ipynb): Hvass, 동영상

- [keras-transfer-learning](https://github.com/neocortex/keras-transfer-learning): 추천, 5개의 노트북으로 구성, TL은 3,4번 노트북

- [deep-photo-styletransfer](https://github.com/luanfujun/deep-photo-styletransfer)

- [Keras Exampls](https://github.com/fchollet/keras/tree/master/examples): fchollet

- [Fine-Tune CNN Model via Transfer Learning](https://github.com/abnera/image-classifier): abnera, [Kaggle](https://www.kaggle.com/abnera/dogs-vs-cats-redux-kernels-edition/transfer-learning-keras-xception-cnn)

* [jupyter_nception v3](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/07_Inception_Model.ipynb),[[논문]](http://arxiv.org/pdf/1512.00567v3.pdf),[[New Version]](https://research.googleblog.com/2016/08/improving-inception-and-image.html)

- [Transfer Learning and Fine Tuning for Cross Domain Image Classification with Keras](https://github.com/sujitpal/fttl-with-keras): sujitpal

- [My experiments with AlexNet, using Keras and Theano](https://rahulduggal2608.wordpress.com/2017/04/02/alexnet-in-keras/): [GithubUb](https://github.com/duggalrahul/AlexNet-Experiments-Keras), [reddit](https://www.reddit.com/r/MachineLearning/comments/64cs0a/p_releasing_codes_for_training_alexnet_using_keras/?st=j1d0iu2p&sh=4bf414d8)
  - AlexNet operates on 227×227 images.

- [Keras pretrained models (VGG16 and InceptionV3) + Transfer Learning for predicting classes in the Oxford 102 flower dataset](https://github.com/Arsey/keras-transfer-learning-for-oxford102):

- [How to use transfer learning and fine-tuning in Keras and Tensorflow to build an image recognition system and classify (almost) any object](https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2)

- [Heuritech project](https://github.com/heuritech/convnets-keras)

- [CarND-Transfer-Learning-Lab](https://github.com/paramaggarwal/CarND-Transfer-Learning-Lab)

- 매틀랩: [Transfer Learning Using Convolutional Neural Networks](https://www.mathworks.com/help/nnet/examples/transfer-learning-using-convolutional-neural-networks.html)

- 매틀랩: [Transfer Learning and Fine-Tuning of Convolutional Neural Networks](http://www.mathworks.com/help/nnet/examples/transfer-learning-and-fine-tuning-of-convolutional-neural-networks.html)

---
![](https://i.imgur.com/md7nNiQ.png)

[전이학습]

깃허브 : https://github.com/Transfer-Learning-with-Python

자료 : https://drive.google.com/drive/folders/1JPPMFYlGMl1BbjjMnG2WqakWwBkiHgzP

동영상 : https://www.youtube.com/playlist?list=PLqkITFr6P-oQlgZS4-XNRcYW2FBUPrc0h