from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import src.face as face
import os
import time

def main():
    start_time = time.time()
    # face.Face.export_detection_for_training_data()
    # face.Face.train()
    f = face.Face()
    print("Time to load model: %s" % str(time.time() - start_time))
    predict_time = time.time()
    for i in range(0, 1):
        result = f.predict(os.path.dirname(__file__) + "/../data/prediction/" + str(i) + ".png")
    print("Predict time : %s" % str(time.time() - predict_time))

if __name__ == '__main__':
    main()
