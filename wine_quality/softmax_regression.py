import tensorflow as tf


# Softmax Regression Model
def softmax_regression(x):
    W = tf.Variable(tf.zeros([10, 2]), name="W")
    b = tf.Variable(tf.zeros([2]), name="b")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y, [W, b]
