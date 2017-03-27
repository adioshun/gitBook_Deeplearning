# RNN (Recurrent Neural Networks)

추후 학습 : [모두를 위한 딥러닝 강좌:RNN](https://youtu.be/-SHPG_KMUkQ?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm0)

## 1. 기본 개념 
* Sequence Data 처리 (eg. 음성)
* 현재의 데이터가 과거의 데이터의 영향을 받는다. 
* NN/CNN은 Sequence를 처리 못함

![](/assets/rnn1.PNG)


![](http://i.imgur.com/vBpftAL.png)


### 1.1 Vanilla RNN 
![](http://i.imgur.com/SmXtkHi.png)


### 1.2 LSTM 
- Vanilla RNN도 깊어 지면 학습이 어려워 짐으로 최근에는 LSTM 을 많이 사용 


### 1.3 GRU by Cho (2014)

## 2. 구조 


## 3. 데모 

## 4. Case Study 
![](/assets/list_of_RNN.png)
- One-to-One : Vanilla Neural Networks
- One-to-Many : Image Captioning (Image -> Sequence of words)
- Many-to-One : Sentiment Classification (Sequence of words -> Sentiment)
- Many-to-Many : Machine Translation (Seq of words -> seq of words)
- Many-to-many : Video Classification on frame level