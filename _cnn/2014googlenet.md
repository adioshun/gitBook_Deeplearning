# GoogLeNet


ILSVRC 2014년 대회에서 우승을 한 구조로 구글의 Szegedy 등에 의해서 개발이 되었다.



## 1. 개요 

이전에 비하여 2014년 이후에는 Deep해졌다. 
- 2013 이전 : 10 Layer 미만
- 2014 이후 : GoogLeNet(22 layer), VGGNet(19 layer)

Layer는 깊어 졌지만, “Inception Module” 개념을 도입하여 파라미터 수를 대폭 줄일 수 있게 되었다.

![](http://i.imgur.com/TL8cPq2.png)



## 2. 특징 

### 2.1 Network In Network

>  “Min Lin”이 2013년에 발표한 “Network In Network” 논문 

#### A. Linear Conv. VS. MLP Conv. 

![](http://i.imgur.com/4n8XIcT.png)

기존 CNN 문제  
- Feature 추출시 filter의 특징이 linear하기 때문에 non-linear한 성질을 갖는 feature를 추출하기엔 어려움(??)
- 이를 극복하기 위해 feature-map의 개수를 늘려야 함 
- 연산량이 늘어남 

제안 해결책 : filter 대신에 MLP(Multi-Layer Perceptron)를 사용하여 feature를 추출

MLP 장점 
-  convolution kernel 보다는 non-linear 한 성질을 잘 활용할 수 있기 때문에 feature를 추출할 수 있는 능력이 우수
- 1x1 convolution을 사용하여 feature-map을 줄일 수 있음

> 1x1 convolution은 중요 개념 이므로 [2.2]에서 다시 설명 


#### B. CNN VS. NIN

![](http://i.imgur.com/1oWxvdX.png)
[ MLPconv layer를 3개 사용한 NIN ]

Fully-connected NN 대신에 최종단에 “Global average pooling”을 사용
- 효과적으로 feature-vector를 추출하였기에 추출된 vector 들에 대한 pooling 만으로도 충분
- Average pooling 만으로 classifier 역할을 할 수 있기 때문에 overfitting의 문제를 회피
    - classifier 역할을 하는 FCN은 전체 free parameter 중 90%를 가지고 있음 
    - 많은 파라미터는 오버피팅에 빠질 가능성유발, FCN이 없으므로 오버피팅 문제 해결 

### 2.2 1x1 convolution

참고 : http://laonple.blog.me/220648539191 중간 

## 3. 구조 (구글의 인셉션-Inception)

Local receptive field에서 더 다양한 feature를 추출하기 위해 여러 개의 convolution을 병렬적으로 활용

||기본구조|확장 구조|
|-|-|-|
||![](http://i.imgur.com/MqoQtOS.png)|![](http://i.imgur.com/lEr9DHG.png)|
|구조|1x1 convolution, 3x3 및 5x5 convolution, 3x3 max pooling|1x1 convolution을 cascade 구조로 두고, <br>1x1 convolution을 통해 feature-map의 개수(차원)를 줄임|
|장점|다양한 scale의 feature를 추출하기에 적합한 구조|feature 추출을 위한 여러 scale을 확보하면서도, 연산량의 균형을 맞출 수 있음|
|단점| 3x3과 5x5 convolution은 연산량이 큼||

