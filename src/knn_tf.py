import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

class kNN:
    def __init__(self,
                 data_train_X,
                 data_train_Y,
                 thresholds,
                 k=1):
        self.data_train_X = data_train_X
        self.data_train_Y = data_train_Y
        self.k = k
        self.thresholds = thresholds
        self.nof_class = np.max(data_train_Y) + 1

        self.x_train, self.y_train, self.x_test, self.prediction = self._build()

    def _build(self):
        feature_number = len(self.data_train_X[0])
        x_train = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)
        y_train = tf.placeholder(shape=[None], dtype=tf.float32)
        x_test = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)

        # euclidean distance
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_train, tf.expand_dims(x_test, 1))), axis=2))

        # nearest k points
        top_k_values, top_k_indices = tf.nn.top_k(tf.negative(distance), k=self.k)
        top_k_labels = tf.to_int32(tf.gather(y_train, top_k_indices))
        top_k_thresholds = tf.to_float(tf.reshape(tf.gather(self.thresholds, top_k_labels, axis=1), (-1, self.k)))
        top_k_labels_one_hot = tf.one_hot(top_k_labels, self.nof_class)

        # calculate weight
        top_k_values = tf.negative(top_k_values)
        sigma = .5
        top_k_weight = tf.exp(tf.div(tf.negative(tf.multiply(top_k_values, top_k_values)), sigma))
        top_k_weight = tf.reshape(tf.multiply(tf.to_float(tf.less_equal(top_k_values, top_k_thresholds)), top_k_weight),(-1, 1, self.k))

        weight_by_labels = tf.matmul(a=top_k_weight, b=top_k_labels_one_hot)
        weight_by_labels = tf.reshape(weight_by_labels, (-1, self.nof_class))

        # prediction
        prediction = tf.multiply(
            tf.to_int32(tf.greater_equal(weight_by_labels, tf.tile(tf.reduce_max(weight_by_labels, axis=1, keepdims=True), (1, weight_by_labels.shape[1])))),
            tf.to_int32(tf.greater(weight_by_labels, tf.tile(tf.reduce_min(weight_by_labels, axis=1, keepdims=True), (1, weight_by_labels.shape[1]))))
        )

        return x_train, y_train, x_test, prediction

    def fit(self, data_test_X):
        with tf.Session() as sess:
            prediction = sess.run(self.prediction, feed_dict={self.x_train: self.data_train_X,
                                                              self.x_test: data_test_X,
                                                              self.y_train: self.data_train_Y})
            return prediction

    @staticmethod
    def evaluate(data_train_Y, data_test_Y, prediction):
        mlb = MultiLabelBinarizer()
        mlb.fit([set(data_train_Y)])
        test_labels_onehot = mlb.transform(list(zip(data_test_Y)))
        # print(prediction)
        # print(test_labels_onehot)
        accuracy = np.mean(np.equal(prediction, test_labels_onehot))
        fp = (prediction - test_labels_onehot) * prediction
        fp = np.sum(fp, axis=1)
        fp = fp[fp != 0]
        # fp = (((predictions - test_labels)!=0)*predictions)
        # fp = fp[fp != 0]
        # fp = fp[fp != np.max(test_labels)+1]
        # print(fp)
        print('Accuracy: %.3f' % accuracy)
        print('False Positive: %.3f' % (len(fp)/prediction.shape[0]))