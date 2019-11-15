# Keep and Learn: Continual Learning by Constraining the Latent Space for Knowledge Preservation in Neural Networks

[https://arxiv.org/pdf/1805.10784.pdf](https://arxiv.org/pdf/1805.10784.pdf), 2018



## 한글 정리 

> [Lunit Tech Blog ](https://blog.lunit.io/2018/08/31/keep-and-learn-continual-learning-by-constraining-the-latent-space-for-knowledge-preservation-in-neural-networks/)

#### 개요 
- multi-center learning : 여러 지역(center)에 흩어져 있는 데이터를 프라이버시 이슈등으로 한곳에 모으지 못하고, 한 지역시 돌아 다니면서 학습을 진행

- 순차적 학습 (continual learning)의 경우 catastrophic forgetting 현상이 발생
	- 특히, gradient descent 알고리즘으로 optimization을 수행하는 neural network 에서는 이 문제가 더욱 심각 

### 기존 연구 

- 방법 #1 : Fine-Tuning (FT) : 가장 naive한 방법, 기존 가중치에 현 데이터를 튜닝 
- 방법 #2 : Elastic Weight Consolidation (EWC,딥마인드), 기존 가중치에 중요도 정의 후 많이 변하지 못하도록 regularization 하는 기법
- 방법 #3 : Learning without Forgetting (LwF), Output activation을 한정하는 방식
	- Original LwF : multi-task multi-center learning
	- LwF+ : single-task multi-center learning

> EWC와 LwF는 complementary하기도 동시 활용 가능 