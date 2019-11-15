# Lifelong Learning

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

## 목적

- 오프라인 러닝과 동일한 성능 `the main objective of an online machine learning algorithm is to try to perform as closely to the corresponding offline algorithm as possible`

## 용어



- Fine-Tuning : 가장 Naive 한 방법, 

- transfer learning

- sequential/Online/Incremental learning : model learns one observation at a time
    - sequential Vs. incremental = 데이터에 순서 법칙이 존재 할때 oder Vs. 데이터에 순서 법칙이 없을때 random
    - online Vs. incremental = Label정보 없음, 이전 내용을 잊을수 있음(Catastrophic Interference) Vs. 라벨 정보 있음, 이전 내용을 잊을 없음
    - online Vs. incremental = faster than the sampling rate VS. runs slower than the sampling rate(updating every 1000 samples)

- Multi-center Learning : 여러 지역(center)에 흩어져 있는 데이터를 프라이버시 이슈등으로 한곳에 모으지 못하고, 한 지역시 돌아 다니면서 학습을 진행 (eg. Incremental Learning과 유사??) 


- Incremental learning : transferring knowledge acquired on old tasks to the new ones,  new classes are learned continually, 얼굴인식 -> 얼굴 식별 


- Lifelong learning is akin to transferring knowledge acquired on old tasks to the new ones. 

- Never-ending learning, on the other hand, focuses on continuously acquiring data to improve existing classifiers or to learn new ones.




> [참고](https://datascience.stackexchange.com/questions/6186/is-there-a-difference-between-on-line-learning-incremental-learning-and-sequent)

---
- [Online Learning](https://www.slideshare.net/queirozfcom/online-machine-learning-introduction-and-examples?from_action=save) : PPT, 60pages

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

- [[MIT 6.883] Online Methods in Machine Learning](http://www.mit.edu/~rakhlin/6.883/)


- [[코세라] Online Learning](https://www.coursera.org/lecture/machine-learning/online-learning-ABO2q): 13min

- [[CSE599s] Online Learning](https://courses.cs.washington.edu/courses/cse599s/12sp/index.html): 2012


---

## Catastrophic Forgetting

- [OVERCOMING CATASTROPHIC FORGETTING FOR CONTINUAL LEARNING VIA MODEL ADAPTATION](https://openreview.net/pdf?id=ryGvcoA5YX): ICLR 2019

- [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796): 2017, [[code]](https://researchcode.com/code/2217672712/overcoming-catastrophic-forgetting-in-neural-networks/)

- [Overcoming Catastrophic Forgetting by Incremental Moment Matching](https://papers.nips.cc/paper/7051-overcoming-catastrophic-forgetting-by-incremental-moment-matching.pdf): NIPS2017, SNU, NAVER

- [Measuring Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1708.02072) : 2017

- [Keep and Learn: Knowledge Preservation in Neural Networks](https://arxiv.org/abs/1805.10784): 2018, [[한글정리]](https://blog.lunit.io/2018/08/31/keep-and-learn-continual-learning-by-constraining-the-latent-space-for-knowledge-preservation-in-neural-networks/)

#### 지속 학습 방법들 


![](https://i.imgur.com/GG74tpz.png)
![](https://i.imgur.com/oPnm1rU.png)



- https://www.slideshare.net/JinwonLee9/deep-learning-seminarsnu161031 : ppt

    - Less forgetting learning in Deep Neural Networks heechul jung 
    - [LwF] Li, Z. and Hoiem, D., Learning without forgetting, In: ECCV (2016)
