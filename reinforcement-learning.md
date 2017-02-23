## 1. 개요 
### 1.1 역사
1. 2013년 Atari BreakoutGame에 적용하면서 인기
2. 2015년 human level control
3. 2016년 AphaGo

## 2. Q-Learning (for deterministic world)
Q(state, Action)을 이용한 문제 해결 

* $$ Max_a {Q(S_1,a)} = Q가 a를 바꿈으로써 얻을수 있는 최대값 $$
* $$ argMax_a{(S_1,a)} = Q가 최대값이 되게 하는 변수 a의 값 $$

### 2.1 기본 알고리즘 
* $$ Q(S,a) = r + Max_{a^1}Q(S^1, a^1)  $$

![](/assets/Screenshot from 2017-02-22 13-54-25.png)
![](/assets/Screenshot from 2017-02-22 14-02-31.png)


### 2.2 기본 알고리즘 + E&E
E&E(Exploit-현재값 이용 & Exploration-탐험적 도전찾음)
 * Dummy Q-Learning의 `Select an Action "a" and execute it`의 a방법
 * 새로운 길을 찾는 기능 추가 
 * ex: 맛집 찾기, 평일은 아는곳(Exploit), 주말은 새로운 곳(Exploration)

###### 구현 방법 1 : E-greedy
 1. `e=0.1 # 10%의 새로운곳 가기`
 2. 랜덤값을 뽑아서 e보다 작으면 새로운곳, e보다 크면 가던곳 

###### 구현 방법 2 : Decaying E-greedy 
 1. `e= 0.1 / (i+1) # for i in range (1000)`
 2. 학습 초기에는 랜덤 비율이 높게, 학습 후반으로 갈수록 랜덤 비율이 작게
![](/assets/decaying_E-greedy.png)

###### 구현 방법 3 : Add random noise 
 1. `a=argmax(Q(s,a) + random_values)`

###### 구현 방법 4 : Add random noise with decaying
 1. `a=argmax(Q(s,a) + random_values/(i+1)) # for i in range (1000)`

> E-greedy방식 대비 Random의 장점은 앞의 Q(s,a)의 값이 고려 되어 선택이 가능 

### 2.2 기본 알고리즘 + E&E + Discount Future Reward
![](/assets/Discount_future_reward.png)
현재 방법은 위 그림 중 (1)과 (2)중 어느것이 좋은줄 판단 어려움 (2가 좋음)

###### 구현 방법 : Discount Future Reward
 1. 현재 받는것은 값을 크게 받고, 이후에 받는같은 $$\gamma$$을 곱해서 받고, 그 이후는 $$\gamma$$의 $$\gamma$$을 곱해서 받음

* $$ Q(S,a) = r + \gamma * Max_{a^1}Q(S^1, a^1)  $$

> 김성훈 교수 설명 [Youtube](https://youtu.be/MQ-3QScrFSI?t=13m44s)


## 3. Q-Learning for Non-deterministic(=Stochastic) world
미끄러운 얼음판 처럼 목적대로 움직이지 않는 경우 

### 3.1 기본 알고리즘 + Learning rate
* Q의 의견은 참조(=Learning rate,$$\alpha$$)만 하고, 내가 가고 싶은 방향($$1-\alpha$$)으로 간다. 

* $$ Q(S,a) = (1-\alpha) * Q(S,a) + \alpha *  [r + Max_{a^1}Q(S^1, a^1)]  $$

## 4. Q-Network
* Q-learning의 경우 Table형태의 작은 배열크기는 가능하지만, 픽셀을 인지하여 겜임을 하는 경우는 크기가 커져 사용이 어려움
* Q -Network를 이용하여 해결 가능 - 딥마인드의 Atari게임 핵심 알고리즘 

### 4.1 네트워크 모델 1
![](/assets/qNet1.png)
* S 와 a를 주고 맞는 Q Value를 도출

### 4.3 네트워크 모델 2
![](/assets/qNet2.png)
* S를 주고 a에 따른 Q value를 도출 

* Linear Regression의 방식과 비슷 ,Cost함수만 변경
$$ 
Cost(w) = (Ws-y)^2 $$일때 $$ y= r + \gamma maxQ(s`) 
$$

> Q Network는 Convergence가 안되고 Diverges(학습이 안됨)됨<-???
> * 이유 1 : Correlation between samples
> * 이유 2 : Non-Stationary Targets
> 
> 이를 해결한 알고리즘이 __DQN__임 

* Tensorflow Q network 코드 설명 [[Youtube]](https://youtu.be/Fcmgl8ow2Uc?list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG)


