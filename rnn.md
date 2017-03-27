
# RNN (Recurrent Neural Networks)
![](https://cdn-images-1.medium.com/max/800/1*laH0_xXEkFE0lKJu54gkFQ.png)

출처 
- [모두를 위한 딥러닝 강좌:RNN](https://youtu.be/-SHPG_KMUkQ?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm0)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/): colah, [번역](https://brunch.co.kr/@chris-song/9)

## 1. 기본 개념 
RNNs의 목적은 배열 (시계열) 데이터를 분류하는 것이라고 말씀드렸습니다. 그리고 RNNs의 학습은 다른 인공 신경망의 학습과 마찬가지로 오차의 backprop과 경사 하강법(Gradient Descent)을 사용합니다.
* Sequence Data 처리 (eg. 음성)
* 현재의 데이터가 과거의 데이터의 영향을 받는다. 
* NN/CNN은 Sequence를 처리 못함

![](/assets/rnn1.PNG)

RNNs은 backprop의 확장판인 [BTPP(Backpropagation Through Time)](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2015/pdfs/Werbos.backprop.pdf)을 사용해 계수를 학습합니다. 
- 본질적으로 BTPP는 기본적인 backprop과 똑같습니다. 
- 다만 RNNs의 구조가 시간에 따라 연결되어 있기 때문에 backprop역시 시간을 거슬러 올라가며 적용되는 것 뿐입니다.

## 2. RNN 종류 
### 2.1 Vanilla RNN 
![](http://i.imgur.com/SmXtkHi.png)

$$ h_t = \tanh(W(h_{t-1}, x_t)+b) $$
- $$h$$ = 현재의 은닉 상태
- $$x$$ = 현재의 입력 

### 2.2 LSTM(Long short term memory)

![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

RNNs의 변형인 `LSTM(Long Short-Term Memory)` 유닛은 90년대 중반에 처음으로 등장했습니다.

LSTM은 오차의 그라디언트가 시간을 거슬러서 잘 흘러갈 수 있도록 도와줍니다
- Vanilla RNN도 깊어 지면 학습이 어려워 짐으로 최근에는 LSTM 을 많이 사용 

셀상태라는 새로운 연산값 필요
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

##### D. 출력 게이트 / 최종 출력값 결정 
![](https://t1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/IgT/image/31zZPrqx8Q8-JTRE_gWuFGtVgGE.png)

- 필터링된 셀 상태 값 


## 정리 
![](http://i.imgur.com/nHQGkpq.png)

$$ f_t = C_{t-1} \times sigmoid(p_t) $$
- C = 셀상태 ($$ f_t + i_t $$)
- $$ p =  W(h_{t-1}, x_t)+b $$

$$ i_t = sigmoid(p_i) \times tanh(p_j) $$

$$ h_t(은닉상태) = tanh(c_t) \times sigmoid(p_o) $$

## 3. RNN Regularizations (오버피팅 문제)
### 3.1 
![](http://i.imgur.com/2MP1BaQ.png)

드랍 아웃 기법 적용 
- 순환 신경망의 출력 값의 흐름 중 `수직`방향에 대해서만 드랍아웃을 적용
- [조심] 순환되는 데이터에는 드랍아웃을 적용하지 않음 

### 3.2 

추천 : [꼭 읽어 보기](http://nmhkahn.github.io/RNN-Regularizations)


## 4. 응용
![](/assets/list_of_RNN.png)
- One-to-One : Vanilla Neural Networks
- One-to-Many : Image Captioning (Image -> Sequence of words)
- Many-to-One : Sentiment Classification (Sequence of words -> Sentiment)
- Many-to-Many : Machine Translation (Seq of words -> seq of words)
- Many-to-many : Video Classification on frame level

##### [Tip] LSTM 하이퍼파라미터 정하기
LSTM의 하이퍼파라미터를 정하는 팁을 몇 가지 적어놓았으니 참고하십시오.
- 과적합(overfitting)이 일어나는지를 계속 모니터링하십시오. 과적합은 신경망이 학습 데이터를 보고 패턴을 인식하는 것이 아니라 그냥 데이터를 외워버리는 것인데 이렇게 되면 처음 보는 데이터가 왔을 때 제대로 결과를 내지 못합니다.
- 학습 과정에서 규제(regularization)가 필요할 수도 있습니다. l1-규제, l2-규제, 드롭아웃을 고려해보십시오.
- 학습엔 사용하지 않는 시험 데이터(test set)를 별도로 마련해두십시오.
- 신경망이 커질수록 더 많고 복잡한 패턴을 인식할 수 있습니다. 그렇지만 신경망의 크기를 키우면 신경망의 파라미터의 수가 늘어나게 되고 결과적으로 과적합이 일어날 수 있습니다. 예를 들어 10,000개의 데이터로 수백만개의 파라미터를 학습하는 것은 무리입니다.
- 데이터는 많으면 많을수록 좋습니다.
- 같은 데이터로 여러 번 학습을 시켜야합니다.
- 조기 종료(early stopping)을 활용하십시오. 검증 데이터(validation set)를 대상으로 얼마나 성능이 나오는지 확인하면서 언제 학습을 끝낼 것인지를 정하십시오.
- 학습 속도(running rate)를 잘 설정하는 것은 정말 너무너무 중요합니다. DL4J의 ui를 써서 학습 속도를 조절해보십시오. 이 그래프를 참고하십시오.
- 다른 문제가 없다면 레이어는 많을수록 좋습니다.
- LSTM에서는 하이퍼탄젠트보다 softsign함수를 사용해보십시오. 속도도 더 빠르고 그라디언트가 평평해지는 문제(그라디언트 소실)도 덜 발생합니다.
- RMSProp, AdaGrad, Nesterove’s momentum을 적용해보십시오. Nesterove’s momentum에서 시작해서 다른 방법을 적용해보십시오.
- 회귀 작업에서는 데이터를 꼭 정규화하십시오. 정말 중요합니다. 또, 평균제곱오차(MSE)를 목적 함수로 하고 출력층의 활성함수(activation function)은 y=x(identity function)을 사용하십시오. (역자 주: 회귀 작업이라도 출력값의 범위를 [0,1]로 제한할 수 있다면 binary cross-entropy를 목적 함수로 사용하고 출력층의 활성함수는 sigmoid를 사용하십시오.)

![](https://i.imgur.com/kpZBDfV.gif)