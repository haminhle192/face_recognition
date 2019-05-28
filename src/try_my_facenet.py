from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import src.face as face
import os
import src.knn_classifier as classifier
import src.encoder1 as encoder1
import time

def main():
    # start_time = time.time()
    # face.Face.export_detection_for_training_data()
    # face.Face.train()
    # f = face.Face()

    # encoder1.Encoder().export_embeddings("E:\Data\Test2", "E:\Data\embeddings3.npy", "E:\Data\labels3.npy",  "E:\Data\label1_string3.npy")
    knn = classifier.KNNClassifier()
    # knn.split_data("E:\Data\embeddings2.npy", "E:\Data\labels2.npy", "E:\Data\\train_embeddings2.npy", "E:\Data\\test_embeddings2.npy")
    knn.test_performance("E:\Data\\train_embeddings3.npy", "E:\Data\\test_embeddings3.npy")
    # knn.test_performance("E:\Data\CASIA_train_embeddings.npy", "E:\Data\CASIA_test_embeddings.npy")

    # print("Time to load model: %s" % str(time.time() - start_time))
    # predict_time = time.time()
    # for i in range(0, 1):
    #     result = f.predict(os.path.dirname(__file__) + "/../data/prediction/" + str(i) + ".png")
    # print("Predict time : %s" % str(time.time() - predict_time))

if __name__ == '__main__':
    main()
