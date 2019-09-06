# Learning Rate 

- Normal Learning Rate :0.01

- Best LR for Adam : 3e-4 - 0.003 (by Andrej Karpathy)



## Annealing(Decay) the Learning Rate 

정의 : 고정된 학습 rate가 아니라 결과에 따라서 동적 변경 

목적 : cost가 떨어지지 않고 정체될경우 LR를 조정 하여 성능 향상 

방법 
- Setp Decay : 
- Exponential Decay 
- 1/t Decay : 

```python 
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)

"""
- starter_learning_rate : 최초 학습시 사용될 learning rate (0.1로 설정하여 0.96씩 감소하는지 확인)
- global_step : 현재 학습 횟수
- 1000 : 곱할 횟수 정의 (1000번에 마다 적용)
- 0.96 : 기존 learning에 곱할 값
- 적용유무 decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
"""
```
