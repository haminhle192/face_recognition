from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import src.facenet as facenet
import os
import pickle
import src.encoder1 as encoder
import src.kNN as kNN
import src.knn_tf as kNNTF
import pandas as pd
import numpy as np
import time
import math
import sys
import src.distance as distance

class KNNClassifier:
    def __init__(
            self,
            data_dir=os.path.dirname(__file__) + "/../data/training_aligned",
            saved_classifier_dir=os.path.dirname(__file__) + "/../saved_classifiers",
            classifier_filename="knn_classifier.pkl",
            n_neighbors=1,
            default_threshold=0.62
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
                # model = kNN.kNN(emb_array, labels, thresholds, self.n_neighbors)
                model = kNNTF.kNN(emb_array, labels, thresholds, sess, k=1)

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
        i = -1
        for n, values in df:
            print(n)
            # st = time.time()
            d = pd.DataFrame(values)
            d = d.sample(frac=1).reset_index(drop=True)
            len = d.shape[0]
            if len >= 20:
                i += 1
                # print(i)
                temp_d = pd.DataFrame(d).drop(labels=[512], axis=1)
                temp_d.insert(512, 512, i)
                train_df = train_df.append(temp_d.iloc[0:10], ignore_index=True)
                test_df = test_df.append(temp_d.iloc[10:len], ignore_index=True)
            else:
                # train_df = train_df.append(d.iloc[0:len//2], ignore_index=True)
                # test_df = test_df.append(d.iloc[len//2:len], ignore_index=True)
                # print("-1")
                temp_d = pd.DataFrame(d).drop(labels=[512], axis=1)
                temp_d.insert(512, 512, -1)
                test_df = test_df.append(temp_d, ignore_index=True)
            # print(time.time() - st)

        # print(train_df)
        # print(test_df)
        np.save(out1_file, train_df.values)
        np.save(out2_file, test_df.values)

    def test_performance(self, train_file, test_file, batch_size=16):
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
        with tf.Session() as sess:
            self.distance_model = distance.Distance(sess)

            print("============Calculate thresholds===============")
            thresholds = self.cal_thresholds(train_data, train_labels)
            # np.save("E:\Data\CASIA_thresholds.npy", thresholds)
            # thresholds = np.load("E:\Data\CASIA_thresholds.npy")
            # thresholds = np.zeros_like(thresholds) + 0.7
            # print(thresholds)
            print("============End Calculate thresholds===============", time.time() - st)
            st = time.time()
            print("============Test performance===============")
            # model = kNN.kNN(train_data, train_labels, thresholds, self.n_neighbors)
            model = kNNTF.kNN(train_data, train_labels, thresholds, sess, k=1)

            nof_test_data = test_data.shape[0]
            print("--->nof test data<----", nof_test_data)
            nof_batch = int(math.ceil(1.0*nof_test_data / batch_size))
            predictions = np.zeros((nof_test_data, np.max(train_labels) + 1))
            # print(predictions.shape)

            for i in range(nof_batch):
                # start = time.time()
                sys.stdout.write("\r --->progress<---- {}/{}".format(i, nof_batch))
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nof_test_data)
                # print(start_index, end_index)
                data = test_data[start_index:end_index, :]
                # print(data.shape)
                pred = model.predict(data)
                # print(pred.shape)
                predictions[start_index:end_index, :] = pred
                # sys.stdout.write("\r --->time by batch<---- {} {}".format(i, time.time()-start))

        # np.save("E:\Data\CASIA_predictions.npy", predictions)
        # predictions = np.load("E:\Data\CASIA_predictions.npy")
        print("------------->Evaluating<-------------")
        kNNTF.kNN.evaluate(train_labels, test_labels, predictions)
        print("------------->End Evaluating<-------------")
        print("============End Test performance===============", time.time() - st)

    def cal_thresholds(self, embeddings, labels):
        df = pd.DataFrame(embeddings)
        df[512] = labels
        max_label = np.max(labels)
        emb_by_classes = df.groupby(512, as_index=False)
        st = time.time()
        print("----->Start calculate threshold cover all data<-----")
        save_to = "E:\Data\CASIA"
        threshold_class = self.get_threshold(embeddings, self.distance_model, self.default_threshold, is_max=False, save_to=save_to+"\\all.npy") - 0.1
        print("----->End calculate threshold cover all data<-----", time.time() - st)
        # print("=========>threshold_class", threshold_class)
        st = time.time()
        print("----->Start calculate threshold for each class<-----")
        thresholds = np.zeros((1, max_label + 1))
        for cls, group in emb_by_classes:
            sys.stdout.write("\r class {}".format(cls))
            group = group.drop(512, axis=1)
            if np.shape(group.values)[0] > 1:
                thresholds[0][cls] = KNNClassifier.get_threshold(group.values, self.distance_model, self.default_threshold, batch_size=1024, save_to=save_to+"\\"+str(cls)+".npy")
            else:
                thresholds[0][cls] = np.min([threshold_class, self.default_threshold])
        print("----->End calculate threshold for each class<-----", time.time() - st)
        # print(thresholds)
        return thresholds

    @staticmethod
    def get_threshold(embeddings, distance_model, default_threshold=0.5, batch_size=32, is_max=True, save_to=''):
        A = embeddings
        B = embeddings
        nof_emb = embeddings.shape[0]
        nof_batch = int(math.ceil(1.0 * nof_emb / batch_size))
        result = np.zeros((nof_batch,))

        for i in range(nof_batch):
            # start = time.time()
            if i % 10 == 0:
                sys.stdout.write("\r  --->progress<---- {}/{}".format(i, nof_batch))
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nof_emb)
            b = B[start_index:end_index, :]
            distances = distance_model.fit(A, b)
            distances = distances[distances != 0]
            if is_max:
                result[i] = np.max(distances, axis=0)
            else:
                result[i] = np.min(distances, axis=0)
            # print("time:", time.time() - start)

        if is_max:
            result = np.max(result, axis=0)
        else:
            result = np.min(result, axis=0)

        np.save(save_to, result)
        # result = np.load(save_to)
        result = np.min(np.array([result, default_threshold]).flatten())
        return result
