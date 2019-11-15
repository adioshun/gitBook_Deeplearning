# [Continual Lifelong Learning with Neural Networks: A Review](https://arxiv.org/abs/1802.07569)

> [한글 정리 ppt 및 후기s](http://dmqm.korea.ac.kr/activity/seminar/266)











---

### [영문 정리](https://medium.com/@datasciencemilan/continual-lifelong-learning-with-deep-architectures-7f25556fb6c)


The goal of Continual Learning is to [overcome “catastrophic forgetting”](https://arxiv.org/abs/1612.00796), in this way the architecture is able to smoothly update the prediction model using several tasks and data distributions.

> catastrophic forgetting: a phenomenon that happens when deep learning are trained sequentially on multiple tasks and the network loses knowledge achieved in the previous ones because weights that are important for a current task are different in the following one

There are several strategies to figure out this matter, in the talk were explained three:

-Naïve Strategy;

-Rehearsal Strategy;

-Elastic Weight Consolidation Strategy