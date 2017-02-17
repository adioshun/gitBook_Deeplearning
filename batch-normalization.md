













---
> 작년에 뉴럴넷 레이어 수를 처음으로 5개 이상으로 시도 해 봤을때 학습이 진행이 잘 안 되거나 에러가 뜨고 죽어 버리는 문제 때문에 한참 고민한 적이 있었다.
그때 Batch Normalization(S. Ioffe and C. Szegedy, 2015)을 알게 되었고 처음 이것(20줄 짜리 간단한 코드)을 적용 해 봤는데 학습 에러(Gradient 폭발)가 없어진 것은 물론이고 학습 결과(정확도)가 더 좋을 뿐 아니라 학습 속도가 10배 이상 빨라져서 짜릿했던 기억이 아직 생생하다.
게다가 이런저런 Weight Initialization 방법들 및 Dropout 등의 Regularization 방법들도 거의 다 필요 없어져 버려서 정말 편했었다.
한마디로 나에게 BN은 Silver Bullet이었다.
그런데 엊그제 Batch Renormalization(S. Ioffe, 2017)이란 것이 또 나왔다.
기존의 BN은 Mini Batch 크기가 작거나 그 샘플링 방법이 좀 어긋나면 효과적으로 동작 하지 않는다. 나도 그래서 고민이 많다. 그래픽 카드 메모리의 한계 때문에 Mini Batch 크기를 마음껏 늘릴수가 없고, 샘플링은 뭘 어떻게 해야 잘 하는 건지 모르겠다.
그런데 이번에 나온 이 Batch Renormalization은 (역시나 간단한 원리로...) 그러한 문제점을 개선 했다고 한다.
TF 구현체도 며칠 내로 나오겠지? 얼른 적용해 보고 싶다. 또 한 번 그 짜릿함을 느낄수 있으려나.
https://arxiv.org/abs/1702.03275
@ 그런데 S. Ioffe 이 분은 정말... 어떻게 이렇게 대단한 발견(또는 발명)을 하고 그걸 또 본인이 남들보다 먼저 발전 시킬수 있는 걸까... 그 실력과 에너지가 엄청나네 정말. 연구 하는 모습을 옆에서 보고 싶네.

---

1. [Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models,2017](https://arxiv.org/abs/1702.03275)

