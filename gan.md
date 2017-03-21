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
