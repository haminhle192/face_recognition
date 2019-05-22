import numpy as np
import sys
import tensorflow as tf


class kNN:
    def __init__(self,
                 data_train_X,
                 data_train_Y,
                 data_test_X,
                 thresholds,
                 k=1):
        self.data_train_X = data_train_X
        self.data_train_Y = data_train_Y
        self.data_test_X = data_test_X
        self.k = k
        self.thresholds = thresholds
        self.nof_class = np.max(data_train_Y) + 1

        sess = tf.Session()
        with sess.as_default():
            self.x_train, self.y_train, self.x_test, self.prediction = self._build()

    # point: n x 1
    def weight(self, distance):
        sigma = .5
        return np.exp(-distance ** 2 / sigma)

    # point: n x 1
    def get_label(self, target):
        max_data_y = np.max(self.data_Y)
        distances = self.get_distance(target)
        # distances = np.append(distances, self.threshold)
        # self.data_Y.append(max_data_y + 1)
        indexes = np.argsort(distances)

        k_neighbors_index = indexes[:self.k]

        # label_weight = np.zeros(max_data_y + 2)
        label_weight = np.zeros(max_data_y + 1)
        for index in k_neighbors_index:
            if distances[index] <= self.thresholds[0, self.data_Y[index]]:
                label_weight[int(self.data_Y[index])] += self.weight(distances[index])

        # print(label_weight)
        if np.count_nonzero(label_weight) > 0:
            return np.argmax(label_weight)
        else:
            return max_data_y + 1

    def _build(self):
        feature_number = len(self.data_train_X[0])
        x_train = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)
        y_train = tf.placeholder(shape=[None, len(self.data_train_Y[0])], dtype=tf.float32)
        x_test = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)

        #euclidean distance
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_train, tf.expand_dims(x_test, 1))), axis=2))

        # nearest k points
        top_k_values, top_k_indices = tf.nn.top_k(tf.negative(distance), k=self.k)
        top_k_values = tf.negative(top_k_values)
        top_k_labels = tf.gather(y_train, top_k_indices)
        top_k_thresholds = tf.gather(self.thresholds, top_k_labels)

        top_k_values = tf.multiply(tf.less_equal(top_k_values, top_k_thresholds), top_k_values)
        sigma = .5
        top_k_values = tf.exp(tf.div(tf.multiply(top_k_values, top_k_values), sigma))

        top_k_labels_one_hot = tf.one_hot(top_k_labels, self.nof_class)
        weight_by_labels = tf.matmul(tf.expand_dims(top_k_values, 1), top_k_labels_one_hot)
        weight_by_labels = tf.reshape(weight_by_labels, (None, self.nof_class))

        prediction = tf.argmax(weight_by_labels, axis=1)

        return x_train, y_train, x_test, prediction

    def fit(self, sess):
        return sess.run(self.prediction(), feed_dict={self.x_train: self.data_train_X,
                                                      self.x_test: self.data_test_X,
                                                      self.y_train: self.data_train_Y})

    def evaluate(self, data_test_Y, prediction):
        accuracy = 0
        for pred, actual in zip(prediction, data_test_Y):
            if pred == np.argmax(actual):
                accuracy += 1

        print(accuracy / len(prediction))

# a = np.ones((2, 4))
# b = np.expand_dims(a, 1)
# c = np.zeros((3, 4))
# d = np.expand_dims(np.sum(np.square(np.subtract(c, b)), axis=2), 1)
# print(d)
# d = tf.constant(d, dtype=tf.float32)
# e = tf.one_hot(tf.constant([[0, 1, 2], [3, 4, 5]]), 6)
# print(d)
# print(e)
# s = tf.matmul(d, e)
# s = tf.reshape(s, (2, 6))
# print(s)
# t = tf.zeros((1, 5))
# t = tf.reduce_max(t, axis=1)
# t = tf.Session().run(t)
# print(t)

from sklearn import preprocessing
