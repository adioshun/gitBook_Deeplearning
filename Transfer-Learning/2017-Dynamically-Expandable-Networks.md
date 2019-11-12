# [Lifelong Learning with Dynamically Expandable Networks](https://arxiv.org/pdf/1708.01547v2.pdf)

 
 > 2017. 울산과기대





---



# [영문 정리 ](https://hackernoon.com/dynamically-expandable-neural-networks-ce75ff2b69cf)

- 정의 : able to learn continuously over time.
-  원리 : techniques like transfer learning are used
	- where the model is trained on previous data
	- and some features are used from that model to learn new data. 

효과 
- This is usually done to reduce the time required to train models from scratch. 
- It is also used when the new data is sparse.


## 1. 구현 방법 - 간단버젼 
- 방법 : constantly fine-tuning the model based on newer data. 
### 문제점 # 1

- However, if the **new task** is very **different** from the old tasks, the model will not be able to perform well on that new task, as features from the old task are not useful, 
	- e.g. if a model that is trained on a million images of animals, it will probably not work very well if it is fine-tuned on images of cars.
### 문제점 # 2 

- after **fine-tuning**, the model may begin to perform the **original** task **poorly** (in this example, predicting animals). 
	- For example, the stripes on a zebra has a vastly different meaning than a striped T-shirt or a fence. Fine-tuning such a model will degrade its performance recognizing zebras.

## 2. 구현 방법 - Dynamically Expandable Networks



- 원리 
	- Train a model, 
	- and if it cannot predict very well, increase its capacity to learn. 
	- If a new task arrives that is vastly different from an existing task, extract whatever useful information you can from the old model and train a new model.


### 2.1 Techniques 

1.  **Selective retraining** — Find the neurons that are relevant to the new task and retain them.
2.  **Dynamic Network Expansion** — If the model is unable to learn from step 1  _(i.e. the loss is above a threshold value)_, increase the capacity of the model by adding more neurons.
3.  **Network Split/Duplication** — If some new models’ units have begun to change drastically, duplicate those weights, and retrain those duplicates, while keeping the old weights fixed.


![](https://i.imgur.com/FgPSPV4.png)
In the above, figure **_t_** denotes task number. Thus, **_t-1_** denotes the previous task, and **_t_** denotes the current task.


#### A. Selective retraining





