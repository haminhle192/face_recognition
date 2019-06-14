from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os

import cv2
import numpy as np

import src.encoder1 as encoder
import src.detection as detection
import src.knn_tf as kNNTF
import time
import tensorflow as tf


class Recognition:
    def __init__(
            self,
            classifier_filename=os.path.dirname(__file__) + "/../saved_classifiers/knn_classifier.pkl",
            debug=False
    ):
        with tf.Session() as sess:
            self.detection = detection.Detection()
            self.identifier = Identifier(classifier_filename, sess)
            self.debug = debug

    def identify(self, image):
        # st = time.time()
        faces = self.detection.find_faces(image)
        # print("dectect time: %s" % str(time.time() - st))
        predicted_faces = []

        for i, face in enumerate(faces):
            if self.debug:
                cv2.imshow("Face: " + str(i), face.image)
            predicted_faces.append(self.identifier.identify(face))

        return predicted_faces


class Identifier:
    def __init__(self, classifier_filename, sess):
        self.encoder = encoder.Encoder(sess=sess)
        print("Loading classifier ...")
        if os.path.exists(classifier_filename):
            with open(classifier_filename, 'rb') as infile:
                emb_array, labels, thresholds, self.class_names = pickle.load(infile)
                self.model = kNNTF.kNN(emb_array, labels, thresholds, k=1)
            print('Loaded classifier model from file "%s"\n' % classifier_filename)
        else:
            print("Don't have classifier! %s" % classifier_filename)

    def identify(self, face):
        with tf.Session() as sess:
            # st = time.time()
            face.embedding = self.encoder.generate_embedding(sess, face)
            # print("embedding time: %s" % str(time.time() - st))
            # st = time.time()
            if face.embedding is not None:
                prediction = self.model.predict(sess, np.array([face.embedding]))[0]
                face.name = self.class_names[prediction[0]]
                # print("recognition time: %s" % str(time.time() - st))
                return face

            else:
                print("Don't have embedding.")
                return "Error"
