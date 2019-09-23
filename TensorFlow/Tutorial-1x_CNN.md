# TensorFlow 1.x CNN Tutorial 

> [TensorFlow 한글 문서](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/)


---

## 1. 데이터 준비 


---

## 2. 모델 생성 



---

## 3. 실행 하기 [(그래프모드)](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/api_docs/python/client.html) `session.run`

```python 

import tensorflow as tf

input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
intermed12 = tf.add(input1, input2)
intermed23 = tf.add(input2, input3)


ops = {'a': intermed12,
            'b': intermed12,
}



with tf.Session() as sess:
  result1, resutl2 = sess.run([ops['a'],ops['b']] )
  print(result1)
````

### Dynamic input with feed_dic


```python

input1 = tf.placeholder(tf.float32)  # feed_dict={input1:[3.3]
input2 = tf.placeholder(tf.float32)  # feed_dict={input2:[3.3]
input3 = tf.placeholder(tf.float32)  # feed_dict={input3:[3.3]

intermed12 = tf.add(input1, input2)
intermed23 = tf.add(input2, input3)


ops = {'a': intermed12,
            'b': intermed23,
}


with tf.Session() as sess:
  result1, resutl2 = sess.run([ops['a'],ops['b']], feed_dict={input1:[3.3],input2:[2.2],input3:[5.5]} )
  print(result1)

```


## 4. 저장 & 복원 

```python 

saver = tf.train.Saver()
saver.restore(sess, restore_model_path) #복원 

sess.run(...)

save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt")) #저장 


```



## 5. 테스트 

---

###### Placeholder

일종의 자료형, 다른 텐서를 할당하는 것

placeholder의 전달 파라미터는 다음과 같다.

```python 
placeholder(
    dtype,      # 데이터 타입을 의미하며 반드시 적어주어야 한다.
    shape=None, # 입력 데이터의 형태를 의미한다. 상수 값이 될 수도 있고 다차원 배열의 정보가 들어올 수도 있다. ( 디폴트 파라미터로 None 지정 )
    name=None   # 해당 placeholder의 이름을 부여하는 것으로 적지 않아도 된다.  ( 디폴트 파라미터로 None 지정 )
)
```

##### tf.get_variable()과 tf.get_collection()

- tf.get_variable() : 텐서의 저장 공간의 주 형태인 variable을 선언하는 방법
    - `tf.Variable()`가 원래 선언 방법. 하지만, `tf.get_variable()`를 사용하는 것이 좀 더 범용적
    - 이유 : `get_variable()`은 정의된 name filed값과 동일한 텐서가 존재할 경우, 새로 만들지 않고 기존 텐서를 불러들인다.
    
```python 
def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,     #  variable의 소속
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 custom_getter=None):
# 출처: https://eyeofneedle.tistory.com/24 [Technology worth spreading
```

- tf.get_collection() :  collection은 variable의 소속
    - 목적은 해당 variable을 코드의 다른 위치에서 불러오기 위해서
    - tf.get_collection(key)가 실행되면, key의 collection에 속하는 variable들의 리스트가 리턴


- tf.get_collection()사용법 7가지 [[자세히]](https://eyeofneedle.tistory.com/24)





