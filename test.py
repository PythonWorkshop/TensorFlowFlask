# restore trained data
import tensorflow as tf

import sys
import wine_quality.model as model
import json
import os
from form import TestParameterForm



x = tf.placeholder("float", [None, 10])
sess = tf.Session()

with tf.variable_scope("softmax_regression"):
    y1, variables = model.softmax_regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "wine_quality/data/softmax_regression.ckpt")
def simple(x1):
    return sess.run(y1, feed_dict={x: x1})

x1 = [[0.7, 0, 1.9, 0.076, 11, 34, 0.99780, 3.51, 0.56, 9.4],
      [0.65, 0, 1.2, 0.065, 15.0, 21, 0.9946, 3.39, 0.47, 10]]

print(simple(x1))
print(y1)
print(variables)
