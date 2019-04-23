from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os

import cv2
import numpy as np

import src.encoder1 as encoder
import src.detection as detection


class Recognition:
    def __init__(
            self,
            classifier_filename=os.path.dirname(__file__) + "/../saved_classifiers/knn_classifier.pkl",
            debug=False
    ):
        self.detection = detection.Detection()
        self.identifier = Identifier(classifier_filename)
        self.debug = debug

    def identify(self, image):
        faces = self.detection.find_faces(image)
        predicted_faces = []

        for i, face in enumerate(faces):
            if self.debug:
                cv2.imshow("Face: " + str(i), face.image)
            predicted_faces.append(self.identifier.identify(face))

        return predicted_faces


class Identifier:
    def __init__(self, classifier_filename):
        self.encoder = encoder.Encoder()
        print("Loading classifier ...")
        if os.path.exists(classifier_filename):
            with open(classifier_filename, 'rb') as infile:
                self.model, self.class_names = pickle.load(infile)
            print('Loaded classifier model from file "%s"\n' % classifier_filename)
        else:
            print("Don't have classifier! %s" % classifier_filename)

    def identify(self, face):
        face.embedding = self.encoder.generate_embedding(face)
        if face.embedding is not None:
            prediction = self.model.predict(np.array([face.embedding]))[0]
            face.name = self.class_names[prediction]
            return face

        else:
            print("Don't have embedding.")
            return "Error"
