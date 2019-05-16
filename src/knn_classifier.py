from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import src.facenet as facenet
import os
import pickle
import src.encoder1 as encoder
import src.kNN as kNN


class KNNClassifier:
    def __init__(
            self,
            data_dir=os.path.dirname(__file__) + "/../data/training_aligned",
            saved_classifier_dir=os.path.dirname(__file__) + "/../saved_classifiers",
            classifier_filename="knn_classifier.pkl",
            n_neighbors=1,
            threshold=0.7
    ):
        self.n_neighbors = n_neighbors
        self.data_dir = data_dir
        self.saved_classifier_dir = saved_classifier_dir
        self.classifier_filename = classifier_filename
        self.threshold = threshold

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
                classifier_filename_exp = os.path.expanduser(os.path.join(self.saved_classifier_dir, self.classifier_filename))

                # Train classifier
                print('Training classifier')
                model = kNN.kNN(emb_array, labels, self.n_neighbors, self.threshold)

                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in data_set]
                class_names.append("-1")

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
