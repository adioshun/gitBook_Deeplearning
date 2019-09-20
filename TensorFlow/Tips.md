# Tensorflow Tips 

## 1. [Eager Execution Mode](https://www.tensorflow.org/guide/eager) [[한글]](https://github.com/tgjeon/TF-Eager-Execution-Guide-KR/blob/master/guide.md)



```
import tensorflow as tf

tf.enable_eager_execution()
print("Eager Mode: ",tf.executing_eagerly())
```




```python 
is_training_pl = tf.placeholder(shape=(),dtype=tf.bool, name="b")
is_training_pl = tf.Variable(tf.zeros(shape=(),dtype=tf.bool), name="b")
```

> https://www.tensorflow.org/beta/guide/migration_guide?hl=ko#%EB%B3%80%ED%99%98_%EC%A0%84


- [Eager 모드 실행을 위해 경할 코드 8가지](https://medium.com/coinmonks/8-things-to-do-differently-in-tensorflows-eager-execution-mode-47cf429aa3ad) : Placeholders, Variables, 

---



## 2. `ft.py_func`


Python 코드를 텐서플로우에서 실행 하는 방법 [[참고]](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/api_docs/python/script_ops.html)

---
## 하이레벨 API

> 출처 : [텐서플로우 하이레벨 API](http://bcho.tistory.com/1195), [Estimator를 이용한 모델 정의 방법](http://bcho.tistory.com/1196)

- tf.contrib: 공식 텐서플로우의 하이레벨 API

- Keras : 공식 하이레벨 API로 로 편입

## 1. Estimator

![](http://cfile30.uf.tistory.com/image/9910C53359AF8CA334DC82)

Estimator: 학습(Training), 테스트(Evaluation), 예측(Prediction)을 한후, 학습이 완료된 모델을 저장(Export)하여 배포 단계를 추상화 한것 

Estimator는
- 직접 개발자가 모델을 직접 구현하여 Estimator를 개발할 수 도 있고 (Custom Estimator)
- 또는 이미 텐서플로우 tf.contrib.learn에 에 미리 모델들이 구현되어 있다.

### 1.1 Estimator 예제

https://github.com/bwcho75/tensorflowML/blob/master/HighLevel%20API%201.%20Linear%20Regression%20Estimator.ipynb



- [새로운 텐서플로 개발 트랜드 Estimator](http://chanacademy.tistory.com/33)


---





