## 1. 개요 
### 1.1 역사
1. 2013년 Atari BreakoutGame에 적용하면서 인기
2. 2015년 human level control
3. 2016년 AphaGo

## 2. Q-Learning
###### Q(state, Action)을 이용한 문제 해결 

* $$ Max_a {Q(S_1,a)} = Q가 a를 바꿈으로써 얻을수 있는 최대값 $$
* $$ argMax_a{(S_1,a)} = Q가 최대값이 되게 하는 변수 a의 값 $$

###### 기본 알고리즘 
* $$ Q(S,a) = r + Max_{a^1}Q(S^1, a^1)  $$

![](/assets/Screenshot from 2017-02-22 13-54-25.png)
![](/assets/Screenshot from 2017-02-22 14-02-31.png)


###### 기본 알고리즘 + E&E
* E&E(Exploit-현재값 이용 & Exploration-탐험적 도전찾음)
 * 새로운 길을 찾는 기능 추가 
 * ex: 맛집 찾기, 평일은 아는곳(Exploit), 주말은 새로운 곳(Exploration)

* 구현 방법 1 : E-greedy
 1. `e=0.1 # 10%의 새로운곳 가기`
 2. 랜덤값을 뽑아서 e보다 작으면 새로운곳, e보다 크면 가던곳 

* 구현 방법 2 : Decaying E-greedy 
 1. `e= 0.1 / (i+1) # for i in range (1000)`
 2. 학습 초기에는 랜덤 비율이 높게, 학습 후반으로 갈수록 랜덤 비율이 작게

* 구현 방법 3 : Add random noise 
 1. `a=argmax(Q(s,a) + random_values)`

* 구현 방법 4 : Add random noise with decaying
 1. `a=argmax(Q(s,a) + random_values/(i+1)) # for i in range (1000)`

> E-greedy방식 대비 Random의 장점은 앞의 Q(s,a)의 값이 고려 되어 선택이 가능 







