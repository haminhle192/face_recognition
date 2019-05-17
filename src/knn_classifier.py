from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import src.facenet as facenet
import os
import pickle
import src.encoder1 as encoder
import src.kNN as kNN
import pandas as pd
import numpy as np


class KNNClassifier:
    def __init__(
            self,
            data_dir=os.path.dirname(__file__) + "/../data/training_aligned",
            saved_classifier_dir=os.path.dirname(__file__) + "/../saved_classifiers",
            classifier_filename="knn_classifier.pkl",
            n_neighbors=1,
            default_threshold=0.7
    ):
        self.n_neighbors = n_neighbors
        self.data_dir = data_dir
        self.saved_classifier_dir = saved_classifier_dir
        self.classifier_filename = classifier_filename
        self.default_threshold = default_threshold

    def load_train_data(self):
        data_set = facenet.get_dataset(self.data_dir)
        # Check that there are at least one training image per class
        for cls in data_set:
            assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the data set')
        paths, labels = facenet.get_image_paths_and_labels(data_set)
        print('Number of classes: %d' % len(data_set))
        print('Number of images: %d' % len(paths))
        return data_set, paths, labels

    def classifier(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                data_set, image_path, labels = self.load_train_data()
                emb_array = encoder.Encoder(sess=sess).generate_embeddings(image_path)
                thresholds = self.cal_thresholds(emb_array, labels)
                classifier_filename_exp = os.path.expanduser(os.path.join(self.saved_classifier_dir, self.classifier_filename))

                # Train classifier
                print('Training classifier')
                model = kNN.kNN(emb_array, labels, thresholds, self.n_neighbors)

                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in data_set]
                class_names.append("-1")

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

    def cal_thresholds(self, embeddings, labels):
        df = pd.DataFrame(embeddings)
        df[512] = labels
        max_label = np.max(labels)
        emb_by_classes = df.groupby(512, as_index=False)
        threshold_class = self.get_threshold(embeddings, self.default_threshold) - 0.1
        print("=========>threshold_class", threshold_class)
        thresholds = np.zeros((1, max_label + 1))
        for cls, group in emb_by_classes:
            print("========================>", cls)
            group = group.drop(512, axis=1)
            if np.shape(group.values)[0] > 1:
                thresholds[0][cls] = self.get_threshold(group.values, self.default_threshold)
            else:
                thresholds[0][cls] = np.min([threshold_class, self.default_threshold])
        print(thresholds)
        return thresholds

    @staticmethod
    def get_threshold(embeddings, default_threshold=0.5):
        A = embeddings
        B = embeddings

        n_point = A.shape[0]

        A = np.repeat(A[np.newaxis, :, :], n_point, axis=0)
        B = np.repeat(B[:, np.newaxis, :], n_point, axis=1)

        result = A - B

        result = np.linalg.norm(result, axis=2)
        result = result.flatten()
        print(result)
        result = result[result != 0]
        result = np.append(result, [default_threshold], axis=0)
        result = np.min(result)
        return result
