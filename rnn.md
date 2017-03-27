
# RNN (Recurrent Neural Networks)
![](https://cdn-images-1.medium.com/max/800/1*laH0_xXEkFE0lKJu54gkFQ.png)

추후 학습 : [모두를 위한 딥러닝 강좌:RNN](https://youtu.be/-SHPG_KMUkQ?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm0)

## 1. 기본 개념 
* Sequence Data 처리 (eg. 음성)
* 현재의 데이터가 과거의 데이터의 영향을 받는다. 
* NN/CNN은 Sequence를 처리 못함

![](/assets/rnn1.PNG)


## 2. RNN 종류 
### 2.1 Vanilla RNN 
![](http://i.imgur.com/SmXtkHi.png)

$$ h_t = \tanh(W(h_{t-1}, x_t)+b) $$
- $$h$$ = 현재의 은닉 상태
- $$x$$ = 현재의 입력 

### 2.2 LSTM(Long short term memory)

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

- Vanilla RNN도 깊어 지면 학습이 어려워 짐으로 최근에는 LSTM 을 많이 사용 

- 셀상태라는 새로운 연산값 필요
    - 하나의 컨베이어 벨트 같이 전체 체인을 관통 - 큰 변화없이 정보 유지가 가능
    - `게이트`라는 요소를 이용하여 정보를 더하거나 제거 함 

##### A. 삭제 게이트/Forget gate layer$$(f_t)$$
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)

- 이전 셀 상태 값중에서 삭제해야 하는 정보를 학습하기 위한 게이트 
- `시그모이드`($$\sigma$$)레이어로 만들어짐 
    - $$ h_{t-1} 과 x_t$$를 입력으로 받아 시그모이드 출력값이 1이면 유지, 0이면 버림 



    
##### B. 입력 게이트$$(i_t)$$
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png)

- 새로운 정보가 셀 스테이트에 저장 될지를 결정하는 게이트 

1. Step 1: Sigmoid Layer를 이용하여 어떤 값을 업데이트 할지 결정 
2. Step 2: Tanh Layer를 새로운 후보값을 만들어 냄 

##### C. 셀 상태 업데이트 
![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png)

- $$ C_{t-1}$$ 가 삭제 게이트 + 입력 게이트를 지나면서 $$ C_t $$로 업데이트 

##### D. 최종 출력값 결정 
![](https://t1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/IgT/image/31zZPrqx8Q8-JTRE_gWuFGtVgGE.png)

- 필터링된 셀 상태 값 


## 정리 
![](http://i.imgur.com/nHQGkpq.png)

$$ f_t = C_{t-1} \times sigmoid(p_t) $$
- C = 셀상태 ($$ f_t + i_t $$)
- $$ p =  W(h_{t-1}, x_t)+b $$

$$ i_t = sigmoid(p_i) \times tanh(p_j) $$

$$ h_t(은닉상태) = tanh(c_t) \times sigmoid(p_o) $$

######  오버피팅 문제 
![](http://i.imgur.com/2MP1BaQ.png)

드랍 아웃 기법 적용 
- 순환 신경망의 출력 값의 흐름 중 `수직`방향에 대해서만 드랍아웃을 적용
- [조심] 순환되는 데이터에는 드랍아웃을 적용하지 않음 


## 3. 응용
![](/assets/list_of_RNN.png)
- One-to-One : Vanilla Neural Networks
- One-to-Many : Image Captioning (Image -> Sequence of words)
- Many-to-One : Sentiment Classification (Sequence of words -> Sentiment)
- Many-to-Many : Machine Translation (Seq of words -> seq of words)
- Many-to-many : Video Classification on frame level