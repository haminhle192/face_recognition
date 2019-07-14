import numpy as np


class kNN:
    # m is #data
    # n is #feature
    # data_X: m x n (X is feature vector n Dimension)
    # data_Y: m x 1 (Y is label = 0,1,2,... #label - 1)
    def __init__(self,
                 data_X,
                 data_Y,
                 thresholds,
                 k=2):
        self.data_X = data_X
        self.data_Y = data_Y
        self.k = k
        self.thresholds = thresholds

    # point: n x 1
    def get_distance(self, target):
        return np.sqrt(np.sum(np.square(self.data_X - target), axis=1))

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
            if distances[index] <= self.thresholds[self.data_Y[index]]:
                label_weight[int(self.data_Y[index])] += self.weight(distances[index])

        if np.count_nonzero(label_weight) > 0:
            return np.argmax(np.nonzero(label_weight))
        else:
            return max_data_y + 1

    # target is set of data point: (#data x #feature) matrix
    def predict(self, target):
        labels = []
        for i in range(target.shape[0]):
            labels.append(self.get_label(target[i]))

        return np.array(labels)