# restore trained data
import tensorflow as tf

import sys
sys.path.append('mnist')
import model

x = tf.placeholder("float", [None, 10])
sess = tf.Session()

with tf.variable_scope("simple"):
    y1, variables = model.softmax_regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "wine_quality/data/simple.ckpt")
def simple(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


