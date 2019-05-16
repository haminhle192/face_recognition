from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from scipy import misc

import src.detection as detection
import src.knn_classifier as classifier
import src.recognition as recognition


class Face:

    def __init__(self):
        self.recognition = recognition.Recognition()

    @staticmethod
    def export_detection_for_training_data(
        input_dir=os.path.dirname(__file__) + "/../data/raw",
        output_dir=os.path.dirname(__file__) + "/../data/raw_aligned",
    ):
        print('Starting export detection...')
        detection.Detection().export_detection(input_dir, output_dir)
        print('Ended export detection.')

    @staticmethod
    def train(
        data_dir=os.path.dirname(__file__) + "/../data/training_aligned",
        saved_classifier_dir=os.path.dirname(__file__) + "/../saved_classifiers",
        classifier_filename="knn_classifier.pkl",
        n_neighbors=1,
        threshold=0.7
    ):
        print("Starting training ...")
        model = classifier.KNNClassifier(data_dir, saved_classifier_dir, classifier_filename, n_neighbors, threshold)
        model.classifier()
        print("Ended training.")

    def predict(self, image_path):
        print("Starting predict ...")
        if os.path.exists(image_path):
            image = misc.imread(image_path)
            faces = self.recognition.identify(image)
            print('number of faces:', len(faces))
            for f in faces:
                print('predicted class:', f.name)
            return (f.name for f in faces)
        else:
            print("Image isn't exist in %s\n" % image_path)
            return ["ERROR"]

    def predict1(self, image):
        faces = self.recognition.identify(image)
        return faces
