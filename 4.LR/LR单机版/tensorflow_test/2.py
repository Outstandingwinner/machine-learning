import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 20)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)

y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

w_L1 = tf.Variable(tf.random_normal([1, 10]))
b_L1 = tf.Variable(tf.zeros([1, 10]))
wx_plus_b_L1 = tf.matmul(x, w_L1) + b_L1
L1 = tf.nn.tanh(wx_plus_b_L1)

w_L2 = tf.Variable(tf.random_normal([10, 1]))
b_L2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_b_L2 = tf.matmul(L1, w_L2) + b_L2
#prediction = tf.nn.sigmoid(wx_plus_b_L2)
#prediction = tf.nn.relu(wx_plus_b_L2)
prediction = tf.nn.tanh(wx_plus_b_L2)

loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})
    prediction_val = sess.run(prediction, feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_val, 'r-', lw=5)
    plt.show()
