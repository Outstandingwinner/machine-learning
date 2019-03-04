import tensorflow as tf
import numpy as np

x_data = np.random.rand(10)
y_data = x_data * 0.3 + 0.8

b = tf.Variable(0.)
w = tf.Variable(0.)

y = w * x_data + b

loss = tf.reduce_mean(tf.square(y_data - y))

train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        sess.run(train_step)
        if step % 50 == 0:
            print step, sess.run([w, b])


