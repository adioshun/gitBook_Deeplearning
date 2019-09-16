# Tensorflow Tips 


## 1. [Eager Execution Mode](https://www.tensorflow.org/guide/eager) [[한글]](https://github.com/tgjeon/TF-Eager-Execution-Guide-KR/blob/master/guide.md)



```
import tensorflow as tf

tf.enable_eager_execution()
print("Eager Mode: ",tf.executing_eagerly())
```




```python 
in_b = tf.placeholder(dtype=tf.float32, shape=(2))

b = tf.Variable(tf.zeros(shape=(2),dtype=tf.float64), name="b")
```

> https://www.tensorflow.org/beta/guide/migration_guide?hl=ko#%EB%B3%80%ED%99%98_%EC%A0%84


---

- [Eager 모드 실행을 위해 경할 코드 8가지](https://medium.com/coinmonks/8-things-to-do-differently-in-tensorflows-eager-execution-mode-47cf429aa3ad) : Placeholders, Variables, 

