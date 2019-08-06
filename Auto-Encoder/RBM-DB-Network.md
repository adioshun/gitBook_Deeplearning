# 제약 볼츠만 머신 

네트워크 종류 
- 계층형 네트워크 
    - 활용 : 패턴 인식
    - 종류 : CNN

- 상호 결합형 네트워크`(그래프 모형을 기원으로 하는 신경망)`
    - 특징 : 층이 없고 유닛끼리 양방향으로 결합. 유닛이 갖는 값에 의해 어떤 상태를 기억 할수 있음
    - 활용 : 최적화 문제, 노이즈 제거, 연상기억 
        - 불완전한 일부의 데이터로부터 나머지 전체의 데이터를 기억(인식) 해 낼 수 있다
    - 종류 : 홉필드 네트워크(1982), 볼츠만, 제약 볼츠만 

> 연산기억 : 어떤 정보를 다른 정보와 연관하여 지어 기억 하는것 eg. 바나나-노랑



## 1. 홉필드 네트워크 

![](https://i.imgur.com/aXzW4Yy.png)


홉필드가 제안(1982년)한 홉필드 신경망은 계산 요소로서 부호 활성화 함수를 따른 맥클록-피츠뉴런을 사용한다.

### 1.1 부호 활성화 함수
- 부호 함수와 비슷하게 동작한다.
- 뉴런의 가중 입력이 0 보다 작으면 출력은 -1 이고, 0 보다 크면 출력은 +1이다.
- 0이면 출력은 바뀌지 않는다. 뉴런이 이젂 출력이 +1이었든 -1이었든 상관 없이 이전상태로 남는다. 


### 1.4 단점 

- 크로스토크(Crosstalk): 기억할 패턴이 서로 비슷하거나, 패턴수가 많아지면 기억을 어렵게 하는 상호 간섭 발생 

- 국소해에 빠지면 패턴을 잘 기억 시킬수 없음 

- 해결법 : 각각의 유닛을 확률적으로 동작 시키는 **볼츠만 머신** 

###### [참고자료]
- [홉필드 네트워크(Hopfield Network)](http://untitledtblog.tistory.com/7)
- [지능시스템론 9. 인공 신경망](http://blog.daum.net/kimjaehun12/184) : 다층신경망, 홉필드, BAM, SOM
- [홉필드 네트워크](http://www.aistudy.co.kr/neural/hopfield_kim.htm): aistudy
- [Hopfield Network를 이용한 패턴 인식#1](http://secmem.tistory.com/268)
- [Hopfield Network를 이용한 패턴 인식#2](http://secmem.tistory.com/270)
- [Hopfield Network를 Python으로 구 ](http://trampkiwi.blog.me/221012687142)
- [BAM을 이용한 패턴 인식#1](http://secmem.tistory.com/335) : BAM은 Hopfield Network와 비슷, 양방향으로 패턴을 연상할 수 있음

---

## 2. 볼츠만 머신(Boltzman machine, BM)

### 2.1 개요
-  신경망과 시뮬레이티드 어닐링 성질들을 결합시킨 모델, 힌튼 제안(1984년)

- 정의 : 홉필드 망의 일반화된 모델/홉필드 네트워크의 동작 규칙을** 확률적인** 동작 규칙으로 확장
- 특징 : 국소해 문제 해결
    - 홉필드 망이나 역전파 망에서는 그라디언트가 아래(에너지가 감소)로만 향하는 방식으로 학습했지만, 볼츠만 머신에서는 확률적으로 증가(에너지가 증가)하는 경우도 있어서 지역 최소값에서 벗어날 수 있게 되었다.


### 2.2 구조 

![](https://i.imgur.com/bQuFI5M.png)

- 각각 하나의 visible layer와 hidden layer로 구성

> Energy Based Model :  E(x,y)  주어진 데이터(x)에 대해서 가능한 모든 출력변수(y)들의 배열들{(x, y)....}의 에너지,  변수들 사이의 상관관계, 최소화 하는 방향으로 학습 진행 


### 2.3 학습 

이 네트워크를 하나의 에너지 모델로 간주합니다. visible layer와 hidden layer가 학습과정에서 에너지 함수를 이용하여 학습

### 2.4 단점 

- unconstrained connectivity(fully connected) 때문에 weight를 구하는 계산과정이 너무 복잡하고 그로 인해 기계학습이나 추론 분야에서 실제적인 문제를 해결하는데 유용하지 않음

- 해결법 : 연결에 제약을 둠 -> 제약 볼츠만


---

## 3. 제약 볼츠만 (Restricted Boltzmann Machine, RBM)

### 3.1 개요 

- BM에서 층간 연결을 없앤 형태의 모습이다.

- 모델의 층간 연결을 없앰으로써, 얻는 이점으로 NN은 깊어질 수 있었다. 

- RBM은 확률모델의 계산이 Intractable 하기 때문에, 학습시 근사법인 MCMC나 또는 제프리 힌튼 교수가 발견한 CD(Contrastive Divergence)를 사용하는데, RBM의 모델의 간단함에서 오는 추론에 대한 이점은 샘플링 근간의 학습법이 실용적인 문제에 적용되는데 기여했다.

- RBM 은 주어진 입력과 똑 같은 출력을 생성하도록 하는 **오토인코딩** 과제를 수행하는 모델

- 무감독학습 신경망 모델

### 3.2 구조 

![](https://i.imgur.com/RJyQFAw.png)

- 2개층 구조 : 관찰 가능한 가시층(visible layer)과 은닉층(hidden)으로만 구성된다

> 같은 층 내부 노드간의 연결은 없고 층과 층 사이와의 연결 = 제약인 이유 


### 3.3 가중치 구하기 

- 역전파 신경망 : 입력노드 * 가중치 -> 시그노이드 함수 적용 -> 결과 = 출력값
- 제  약 볼츠만 : 입력노드 * 가중치 -> 시그노이드 함수 적용 -> 결과의 확률값 = 출력값

### 3.4 Energy 값

- 역전파 신경망 : 목표치를 기준으로 오류치 계산하여 수정 
- 제  약 볼츠만 : 단층이라 목표치 없음, 홉필드 네트워크와 같은 개념, 입력과 출력이 양의 상관 관계에 있을수록 에너지가 작아진다. 

### 3.5 학습 방법 

1. 입력층 노드값에 가충치를 곱하고 더하고 확률을 구하고 확률에 근거하여 샘플링하여 은닉층의 노드값을 결정

2. 은닉층의 노드값이 결정되면 또 다시 입력층의 노드값들을 같은 방식으로 결정

3. 에너지를 최소화하는 방향으로 가충치를 학습


RBM도 마찬가지로 log-likelihood의 gradient descent를 수행함으로써 네트워크를 학습합니다. 학습 과정에서 Markov chain을 수렴할 때까지 반복함으로써 p(x)로부터의 표본들을 구하는 Gibbs sampling을 사용합니다. 하지만 표본을 추출하기 위해 sampling을 수렴할 때 까지 반복하는 것은 비효율적이기 때문에, 표본 추출 시 속도 개선을 위해 contrastive divergence 방식을 사용합니다.


###### [참고자료]
- [제한 볼츠만 머신 초보자 튜토리얼](http://blog.naver.com/rupy400/220793514761)
- [RBM (제한된 볼츠만 머신) 이해](http://www.tbacking.com/?p=351)
- [제한된 볼츠만 기계 (리스트릭티드 볼츠만 머신, RBM)](http://neuralix.blogspot.com/2014/02/draft.html)
- [Restricted Boltzmann Machine #1 (Korean)](http://junya906.blogspot.com/2016/06/restricted-boltzmann-machine-1-korean.html)
- Asja Fischer, [An Introduction to Restricted Boltzmann Machines](http://image.diku.dk/igel/paper/AItRBM-proof.pdf), CIARP 2012

---

## 4. Deep Belief Network

- RBM은 우선 DBN(DBN, Deep Belief Network)이라고 하는 딥러닝의 일종인 심층신뢰신경망을 구성하는 기본적인 요소이다. 즉 RBM을 여러겹 쌓아서 심층신뢰신경망을 만든다.

- 2006 년에 최초로 컨볼루션 연산을 사용하지 않고 심층 구조 상에서 학습을 성공시킨 모델

- 오차 소멸 문제 해결을 위해 층별 선훈련 (layerwise pre-training)과정 수행 : 아래층에 대해서 가중치를 먼저 학습한 후 이 가중치를 고정한 다음 그 다음 층의 가중치를
학습

- 딥빌리프네트워크는 크게 두 가지로 구분된다. 
    - 하나는 입력과 같은 출력을 재생성하도록 하는 오토인코더 (무감독 학습)
    - 다른 하나는 분류기로 사용하는 것이다.(감독학습)


###### [참고자료] 
- [서울대 강의 자료](https://bi.snu.ac.kr/Courses/ML2016/LectureNote/LectureNote_ch5.pdf): 장병탁교수, 식 유도 과정 설명  