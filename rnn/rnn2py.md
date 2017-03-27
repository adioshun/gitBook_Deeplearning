텐서플로로 구현된 LSTM 쉬운 예제입니다.
간단한 사용법만 잘 숙지하면 이렇게 간단하게 구현할 수 있네요.

이걸 못해서 여태 계속 어려운 코드만 눈빠져라 보고 있었던 시간이 허무해집니다 ㅠㅠ

MNIST 이미지에서 하나의 row를 하나의 데이터로 보고 한 이미지에 들어있는 row들의 집합을 시퀀스로 해석해서 풉니다.

mnist 데이터 28by28이니까 28개의 row가 나오는 것이고 28개의 row 데이터들의 하나의 숫자를 의미하는 시퀀스가 되겠습니다.
그런 식으로 해서 mnist 손글씨 인식을 해내는 코드입니다. 

> 출처 : [모두의 연구소](http://www.modulabs.co.kr/index.php?mid=DeepLAB_free&page=3&document_srl=2107)

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))


x = tf.transpose(x, [1, 0, 2])
x = tf.reshpae(x, [-1, n_input])
x = tf.split(0, n_steps, x )

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell( n_hidden, forget_bias=1.0)
outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
pred = tf.matmul(outputs[-1], weights ) + biases

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(pred, y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        sess.run(train, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print "step : %d, acc: %f" % ( step, acc )
        step += 1
    print "train complete!"

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print "test accuracy: ", sess.run( accuracy, feed_dict={x: test_data, y: test_label})

```