from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Distance:
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.b_tf = tf.placeholder(shape=[None, 512], dtype=tf.float32)
            self.A_tf = tf.placeholder(shape=[None, 512], dtype=tf.float32)
            self.distance_tf = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.A_tf, tf.expand_dims(self.b_tf, 1))), axis=2))

    def __del__(self):
        self.sess.close()

    def fit(self, A, b):
        return self.sess.run(self.distance_tf, feed_dict={self.A_tf: A, self.b_tf: b})
