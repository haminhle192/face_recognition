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
import time


class KNNClassifier:
    def __init__(
            self,
            data_dir=os.path.dirname(__file__) + "/../data/training_aligned",
            saved_classifier_dir=os.path.dirname(__file__) + "/../saved_classifiers",
            classifier_filename="knn_classifier.pkl",
            n_neighbors=1,
            default_threshold=0.901
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

    def split_data(self, data_file, label_file, out1_file, out2_file):
        df = pd.DataFrame(np.load(data_file))
        df[512] = np.load(label_file)
        df = df.sort_values(by=[512]).groupby([512])
        train_df = pd.DataFrame(np.zeros((0, 513)))
        test_df = pd.DataFrame(np.zeros((0, 513)))
        for n, values in df:
            print(n)
            # st = time.time()
            d = pd.DataFrame(values)
            d = d.sample(frac=1).reset_index(drop=True)
            len = d.shape[0]
            if len >= 20:
                train_df = train_df.append(d.iloc[0:4], ignore_index=True)
                test_df = test_df.append(d.iloc[4:len], ignore_index=True)
            else:
                # train_df = train_df.append(d.iloc[0:len//2], ignore_index=True)
                # test_df = test_df.append(d.iloc[len//2:len], ignore_index=True)
                test_df = test_df.append(d, ignore_index=True)
            # print(time.time() - st)

        np.save(out1_file, train_df.values)
        np.save(out2_file, test_df.values)

    def test_performance(self, train_file, test_file):
        st = time.time()
        print("============Loading data===============")
        train_df = pd.DataFrame(np.load(train_file))
        test_df = pd.DataFrame(np.load(test_file))
        train_data = train_df.drop(labels=[512], axis=1).values
        train_labels = np.ndarray.tolist((train_df.values[:, -1]).astype(int))
        test_data = test_df.drop(labels=[512], axis=1).values
        test_labels = np.ndarray.tolist((test_df.values[:, -1]).astype(int))
        print("============End Loading data===============", time.time() - st)
        st = time.time()
        print("============Calculate thresholds===============")
        thresholds = self.cal_thresholds(train_data, train_labels)
        # thresholds = np.zeros_like(thresholds) + 0.7
        print(thresholds)
        print("============End Calculate thresholds===============", time.time() - st)
        st = time.time()
        print("============Test performance===============")
        model = kNN.kNN(train_data, train_labels, thresholds, self.n_neighbors)
        predictions = model.predict(test_data)
        print(predictions)
        print(test_labels)
        accuracy = np.mean(np.equal(predictions, test_labels))
        fp = (((predictions - test_labels)!=0)*predictions)
        fp = fp[fp != 0]
        fp = fp[fp != np.max(test_labels)+1]
        print(fp)
        print('Accuracy: %.3f' % accuracy)
        print('False Positive: %.3f' % (len(fp)/predictions.shape[0]))
        print("============End Test performance===============", time.time() - st)

    def cal_thresholds(self, embeddings, labels):
        df = pd.DataFrame(embeddings)
        df[512] = labels
        max_label = np.max(labels)
        emb_by_classes = df.groupby(512, as_index=False)
        threshold_class = self.get_threshold(embeddings, self.default_threshold) - 0.1
        print("=========>threshold_class", threshold_class)
        thresholds = np.zeros((1, max_label + 1))
        for cls, group in emb_by_classes:
            # print("========================>", cls)
            group = group.drop(512, axis=1)
            if np.shape(group.values)[0] > 1:
                thresholds[0][cls] = KNNClassifier.get_threshold(group.values, self.default_threshold)
            else:
                thresholds[0][cls] = np.min([threshold_class, self.default_threshold])
        # print(thresholds)
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
        print("++++++++++++++++++++++")
        print(result)
        result = result[result != 0]
        result = np.max(result, axis=0)
        result = np.min(np.array([result, default_threshold]))
        return result
