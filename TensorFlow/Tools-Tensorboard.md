![](https://i.imgur.com/CMjq8n4.png)

1. 코드에 텐서보드용 로그 파일 저장 위치 지정
```
writer = tf.train.SummaryWriter("/tmp/test_logs", session.graph)
```

2. Tracking할 대상 지정 

# step 1: node 선택
add_hist = tf.scalar_summary("add_scalar", add)
mul_hist = tf.scalar_summary("mul_scalar", mul)

# step 2: summary 통합. 두 개의 코드 모두 동작.
merged = tf.merge_all_summaries()
# merged = tf.merge_summary([add_hist, mul_hist])


3. 학습 수행시 로그 파일 저장 ` writer.add_summary(summary, step)`


4. 실행 

```
tensorboard --logdir=/tmp/sample
# tensorboard --logdir=/tmp/sample --port=8008
```

# 주요 함수
## 1. tf.name_scope()
- name_scope 함수는 노드의 이름을 지정하고 노드의 큰 틀을 제공해줍니다. 
- 그 틀 안에서 실행되는 연산들은 텐서보드에서 노드를 클릭하면 연산의 흐름을 볼 수 있습니다.
- 블록 단위로 나누어서 표현 하고자 할때

![](https://1.bp.blogspot.com/-INCrGDDl-Ow/V7W1dzBGjGI/AAAAAAAAIGA/wKj5QuDCm1oa_XKL0kgbbXS72cSksO3cgCK4B/s640/ScreenShot_20160812235353.png)
name_scope가 accuracy , cost , layer1,2,3로 총 5번 나옵니다

## 2.  tf.histogram_summary()
![](https://2.bp.blogspot.com/-18Ljre-zZmk/V7W6BVjt1AI/AAAAAAAAIGM/g4FkqSAM7iYZh26pT0xIQUOUnjnHSwW2gCK4B/s640/ScreenShot_20160812235353.png)
- 히스토그램으로 변수를 요약

> summary 함수는 대부분이 `변수`의 변화 양상을 그래프로 보여주는 함수일 것입니다

> 그래디언트 결과물이나 가중치 변수에 [histogram_summary](https://tensorflowkorea.gitbooks.io/tensorflow-kr/g3doc/api_docs/python/train.html#histogram_summary) 작업(op)을 추가해서 데이터를 모을 수 있음

## 3. `tf.scalar_summary()`
![](https://2.bp.blogspot.com/-kT8RsG5nUjE/V7W72y6nrEI/AAAAAAAAIGY/YPuE6LERrbQUxaGDOEHZVP2mGeodJMSngCK4B/s640/ScreenShot_20160812235353.png)
- histogram_summary과 마찬가지로 요약해주는 함수입니다. 
- 하지만 이 함수는 scalar로 변수의 변화를 요약

> 학습률과 손실을 각각 만들어내는 노드에 [scalar_summary](https://tensorflowkorea.gitbooks.io/tensorflow-kr/g3doc/api_docs/python/train.html#summary-operations) 작업(op)을 추가해서 데이터를 모을 수 있음

###### 추가 summary함수들 
[링크](https://tensorflowkorea.gitbooks.io/tensorflow-kr/g3doc/api_docs/python/train.html#summary-operations)

- tf.image_summary(tag, tensor, max_images=3, collections=None, name=None)
- tf.audio_summary(tag, tensor, sample_rate, max_outputs=3, collections=None, name=None)


## 4. `tf.train.SummaryWriter()` and `tf.train.SummaryWriter.add_summary()`
- tf.train.SummaryWriter 클래스는 events file을 log 디렉토리에 생성하고 events와 summaries를 추가하는 함수입니다.
- tf.train.SummaryWriter.add_summary 함수는 tf.train.SummaryWriter 클래스의 한 함수이며, 코드에서는 학습할 때마다 요약을 추가해서 그래프를 만드는 것입니다.

>  [tf.merge_all_summaries](https://tensorflowkorea.gitbooks.io/tensorflow-kr/g3doc/api_docs/python/train.html#merge_all_summaries)를 사용해서 모든 요약 노드들을 하나로 합쳐서 한 번에 모든 요약 데이터를 만들 수 있음

###### 샘플코드 #1 : tensorboard에 점 하나 찍는 예제

```python
import tensorflow as tf

a = tf.constant(3.0)
b = tf.constant(5.0)
c = a * b

# tensorboard에 point라는 이름으로 표시됨
c_summary = tf.scalar_summary("point", c)
merged = tf.merge_all_summaries()

with tf.Session() as sess:
    writer = tf.train.SummaryWriter("./board/sample_1", sess.graph)

    result = sess.run([merged])
    tf.initialize_all_variables().run()

    writer.add_summary(result[0])
```

###### 샘플코드 #2 : 두 개의 직선을 출력하는 예제

```python
import tensorflow as tf

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

add = tf.add(X, Y)
mul = tf.mul(X, Y)

# step 1: node 선택
add_hist = tf.scalar_summary("add_scalar", add)
mul_hist = tf.scalar_summary("mul_scalar", mul)

# step 2: summary 통합. 두 개의 코드 모두 동작.
merged = tf.merge_all_summaries()
# merged = tf.merge_summary([add_hist, mul_hist])

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    # step 3: writer 생성
    writer = tf.train.SummaryWriter("./board/sample_2", sess.graph)

    for step in range(100):
        # step 4: 노드 추가
        summary = sess.run(merged, feed_dict={X: step * 1.0, Y: 2.0})
        writer.add_summary(summary, step)

# step 5: 콘솔에서 명령 실행
# tensorboard --logdir=./board/sample_2
```

###### 샘플코드 #3 : MNIST 예제


---

- [파이쿵](http://pythonkim.tistory.com/39)
- [COMPUTER & BOOKS](http://byungjin-study.blogspot.com/2016/08/tensorboard.html)
- [공식 메뉴얼_한글](https://tensorflowkorea.gitbooks.io/tensorflow-kr/g3doc/how_tos/summaries_and_tensorboard/)
- [공식 홈페이지](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/README.md)

- [Debugging & Visualising training of Neural Network with TensorBoard](https://www.analyticsvidhya.com/blog/2017/07/debugging-neural-network-with-tensorboard/)

- [jupyter-tensorboard](https://github.com/lspvic/jupyter_tensorboard)
