# 네트워크 설계

> [레이어 파라미터 계산 하기(Youtube)](https://www.youtube.com/watch?v=rySyghVxo6U&list=PLQ28Nx3M4JrhkqBVIXg-i5_CVVoS1UzAv&index=19)

가정사항
- 털 과 날개의 유무(feature=2)에 따라, 기타, 포유류,조류(Classificatin=3)을 하는 신경모델 

## 1. 가중치(weight) 설계 


첫번째 가중치의 차원은 2차원으로 [특성, 히든 레이어의 뉴런갯수] -> [2, 10] 으로 정합니다.
```
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
```

마지막 가중치의 차원을 [이전 히든 레이어의 뉴런 갯수, 분류 갯수] -> [10, 3] 으로 정합니다.
```
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))
```

## 2. 편향(Bias)설계 
- 편향을 각각 각 레이어의 `아웃풋 갯수`로 설정합니다.

b1 은 히든 레이어의 뉴런 갯수로 : `b1 = tf.Variable(tf.zeros([10]))`
b2 는 최종 결과값 즉, 분류 갯수인 3으로 설정 : `b2 = tf.Variable(tf.zeros([3]))`


## 3. CNN 설계

참고 : http://128.46.80.28:8585/edit/3_golbin/MNIST_CNN/CNN.py

### 3.1 Conv 네트워크 
#### A. 첫번째 레이어 
tf.nn.conv2d(X, W) = [입력이미지수, 가로, 세로, 필터갯수] = [?, 28, 28, 32]
- X = 입력 이미지 차원 = eg. MNIST경우 28*28*1
- W = 커널크기 `3,3` , 입력값 X의 특성수 `1`, 필터 갯수 `32` = [3, 3, 1, 32]

#### B. 두번째 레이어 
- L2 Conv shape=(?, 14, 14, 64)
    - Pool     ->(?, 7, 7, 64)
    - Reshape  ->(?, 256)
       
W2 의 [3, 3, 32, 64] 에서 32 는 L1 에서 출력된 W1 의 마지막 차원, 필터의 크기 입니다.

### 3.2 Pool 
Pool     ->(?, 14, 14, 32)


### 3.3 Full Connect Layer
- FC 레이어: 입력값 7x7x64 -> 출력값 256

## 4. RNN 설계

```
1. rnn_cell = rnn_cell.BasicRNNCell(rnn_size)
2. state = tf.zeros([batch_size, rnn_cell.state_size])
3. X_split = tf.split(0,time_step_size, x_data)
4. outputs, state = rnn.rnn(rnn_cell, X_split, state)
```


> 신 버젼 변경 

> rnn_cell->tf.nn.rnn_cell -> tf.contrib.rnn

> rnn.rnn->tf.nn.rnn﻿ -> tf.contrib.rnn

### 4.1 셀 정의 
`rnn_cell=tf.nn.rnn_cell.BasicRNNCell(num_units=num_units)`
- tf.RNN에서 hidden layer == output으로 보기 때문에 지금 예시같은 경우에는 
- 여기서 num_units는 그냥 뉴럴넷에서 히든레이어 노드 수라고 보면됩니다, h,e,l,o니까 4개 또는 length입니다

### 4.2 Initial state
`hidden_state_initial=rnn_cell.zero_state(batch_size,dtype=tf.float32)`
- 또한 RNN은 t=0을 시작이라고 했을때 t=-1일때의 initial state가 필요하기 때문에 셀의 크기에 맞춰 다 0으로 초기화 해놓습니다
- 또한 지금 input_x는 4x4 행렬인데 이를 1x4씩 시간에 따라 넣어줘야하기 때문에 tf.split(split_dim, num_split, value, name='split')
    - split_dim 기준으로 num_split 등분해주는 함수입니다


### 4.3 RNN정의 
` output, state = rnn.rnn(rnn_cell, X-Split, state)
- 각셀을 옆으로 몇개(X-Split) 이어서 구성할것인지 정의 (=Time Stamp)

tf.Tensor 'splilt:0 shape=(배치사이즈, 단어수) dtype=float32


### 4.4 Cost 정의 
`sequence_loss_by_example([logits=예측값], [targets=정답], [weights=보통1])`

- logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols].
    - `logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])`


- targets: list of 1D batch-sized int32 Tensors of the same length as logits.
    - `targets = tf.reshape(sample[1:], [-1])`


- weights: list of 1D batch-sized float-Tensors of the same length as logits.
    - `weights = tf.ones([time_step_size * batch_size])`

```
loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)
```
## 5. GAN 설계 
