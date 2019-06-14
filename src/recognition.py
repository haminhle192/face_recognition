from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os

import cv2
import numpy as np

import src.encoder1 as encoder
import src.detection as detection
import time
from scipy import misc
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
        # faces = self.detection.find_faces(image)
        # print("dectect time: %s" % str(time.time() - st))
        # resized_image = misc.imresize(image, (160, 160), interp='bilinear')
        face = detection.Face()
        face.image = image
        faces = [face]
        predicted_faces = []

        for i, face in enumerate(faces):
            predicted_faces.append(self.identifier.identify(face))

        return predicted_faces


class Identifier:
    def __init__(self, classifier_filename, sess):
        self.encoder = encoder.Encoder(sess=sess)
        print("Loading classifier ...")
        if os.path.exists(classifier_filename):
            with open(classifier_filename, 'rb') as infile:
                self.model, self.class_names = pickle.load(infile)
            print('Loaded classifier model from file "%s"\n' % classifier_filename)
        else:
            print("Don't have classifier! %s" % classifier_filename)

    def identify(self, face):
        # st = time.time()
        face.embedding = self.encoder.generate_embedding(face)
        # print("embedding time: %s" % str(time.time() - st))
        # st = time.time()
        if face.embedding is not None:
            prediction = self.model.predict(np.array([face.embedding]))[0]
            face.name = self.class_names[prediction]
            # print("recognition time: %s" % str(time.time() - st))
            return face

        else:
            print("Don't have embedding.")
            return "Error"
