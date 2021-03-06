

## What Is Federated Learning?

> October 13, 2019 by NICOLA RIEKE [[출처]](https://blogs.nvidia.com/blog/2019/10/13/what-is-federated-learning/)

> 모델 받아서 자신의데이터로 학습후 모델만 전달, 반복...
![](https://i.imgur.com/k7wTNG5.png)


- 일부 학습 데이터는 서로 다른 공간에 나누어져 있다. 
    - (eg. 의료 데이터는 프라이버시 이슈로 병원을 벗어 날수 없으며, 각 병원마다 보유 하고 각 병원 데이터로만 학습을 진행 하기에 bias될 가능성이 크다)
- 연합 학습은 다른 공간에 있는 데이터를 학습할수 있도록 해준다. (`Federated learning makes it possible for AI algorithms to gain experience from a vast range of data located at different sites.`)

- 민감한 의료 데이터를 직접 교환하지 않고, 협업하여 모델 성능 개선에 참여 할수 있도록 한다. 

### How Federated Learning Works 

https://www.youtube.com/watch?time_continue=68&v=Jy7ozgwovgg&feature=emb_logo

- the model is trained in multiple iterations at different sites.

- 한곳에서 자신의 데이터로 학습하고 학습된 모델을 연합 서버로 전달, 다른 병원서 해당 모델을 넘겨 받아 자시의 데이터로 학습후 학습된 모델을 다시 연합 서버로 전달...반복 


---

- 엔비디아가 클라라 연합학습(Clara Federated Learning, 이하 클라라 FL)을 최초로 공개

- [연합학습으로 방사선학의 딥러닝 간소화...엣지 컴퓨팅 플랫폼](http://www.aitimes.kr/news/articleView.html?idxno=14810)