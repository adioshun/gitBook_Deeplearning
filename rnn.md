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

$$ h_t = \tanh(W(h_{t-1}, x_t)+b) $$
- $$h$$ = 현재의 은닉 상태
- $$x$$ = 현재의 입력 

### 1.2 LSTM(Long short term memory)
- Vanilla RNN도 깊어 지면 학습이 어려워 짐으로 최근에는 LSTM 을 많이 사용 
- 셀상태라는 새로운 연산값 필요
    - 삭제 게이트$$(f_t)$$: 이전 셀 상태 값중에서 삭제해야 하는 정보를 학습하기 위한 게이트 
    - 입력 게이트$$(i_t)$$: 새롭게 추가 되어야 하는 정보를 학습하기 위한 게이트 

![](https://cdn-images-1.medium.com/max/800/1*laH0_xXEkFE0lKJu54gkFQ.png)

$$ f_t = C_{t-1} \times sigmoid(p_t) $$
- C = 셀상태 ($$ f_t + i_t $$)
- $$ p =  W(h_{t-1}, x_t)+b $$

$$ i_t = sigmoid(p_i) \times tanh(p_j) $$

$$ h_t(은닉상태) = tanh(c_t) \times sigmoid(p_o) $$

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