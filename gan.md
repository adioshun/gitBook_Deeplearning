![](https://cdn-images-1.medium.com/max/800/1*vI-O4LpRsj3Ac-9lRYGxDQ.png)
- GAN방식과 다른 방식의 분류 

### Generative Adversarial Nets

> 출처 : [초짜 대학원생 입장에서 이해하는 Generative Adversarial Nets](http://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-1.html)


# 개요 
저자 : Ian Goodfellow
논문 : [Generative Adversarial Networks, 2014](https://arxiv.org/abs/1406.2661) , [NIPS 2016 Tutorial:
Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160v1.pdf)

## 1. 목적 
> 지폐위조범(Generator)은 경찰을 최대한 열심히 속이려고 하고 다른 한편에서는 경찰(Discriminator)이 이렇게 위조된 지폐를 진짜와 감별하려고(Classify) 노력한다.
이런 경쟁 속에서 두 그룹 모두 속이고 구별하는 서로의 능력이 발전하게 되고 결과적으로는 진짜 지폐와 위조 지폐를 구별할 수 없을 정도(구별할 확률 pd=0.5)에 이른다는 것.

![](https://2.bp.blogspot.com/-2AA2ws2s6xc/WHjzFO5zBbI/AAAAAAAABKw/g91OEkkxPHYKPTsfKYC5yzXn3CmH6gi_ACK4B/s400/%25EA%25B7%25B8%25EB%25A6%25BC6.PNG)


GAN의 가장 큰 약점은 만드는 쪽과 구별하는 쪽을 균형 있게 훈련시키기가 기존의 최적화에 비해 매우 어렵다는 것이다. 

게임을 하는 두 사람 간의 실력 차가 크다면 서로의 발전을 기대하기는 어렵다. GAN의 학습도 이러한 ‘실력 차’에 의한 불균형이 종종 발생해 훈련을 어렵게 하곤 한다.

![](https://3.bp.blogspot.com/-hif7IOq8eW0/WGYUKAIh_II/AAAAAAABWQw/dp7d9sxq_sotgPGie5sDA_rlgxOvPxcMQCLcB/s640/Screen%2BShot%2B2016-12-30%2Bat%2B02.59.36.png)
arg max D: 여기에서는 목적함수를 극대화하는 분류망 D를 찾는다
- 첫번째 항 E[Log D(x)]은 실제 데이터 (x), 예를 들어 진짜 그림을 넣었을 때의 목적함수의 값이다. 
- 두번째 항 E[log(1-D(g(z)))]은 가짜 데이터 (G(z)), 즉 생성망이 만들어낸 그림이 들어가있다. 그리고 arg max D인데 항 내부는 1-D(G(z))이다. 다시 말해 둘째 항의 극대화는 D(G(z))의 극소화다.
- 결과적으로, 이 두 항의 목적함수를 이용해 
    - 진짜 그림을 넣으면 큰 값을,
    - 가짜 그림을 넣으면 작은 값을
- ..출력하도록 구별망 D를 열심히 학습시키자는 것이다.

arg min G: 이 말은 목적함수를 극소화하는 생성망 G를 찾자는 이야기다.
- G는 두번째 항에만 포함되어있다.
- 전체 함수를 극소화하는 G는,  둘째 항을 극소화하는 G이고, 결국 D(G(z))를 극대화하는 G이다.
- 결과적으로, 구별망을 속이는 생성망 G를 열심히 학습시켜보자는 이야기다.

![](https://4.bp.blogspot.com/-2piuSDKxkkk/WGYRZECArDI/AAAAAAABWQU/9Ezk7b1wP8g_BSktQ5yDLUBiB9ZeBhPaACLcB/s400/gan.png)