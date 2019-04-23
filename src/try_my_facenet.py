from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import src.face as face
import os
import time

def main():
    start_time = time.time()
    face.Face.export_detection_for_training_data()
    # face.Face.train()
    # f = face.Face()
    # f.predict(os.path.dirname(__file__) + "/../data/predict/25.png")
    print("Time: %s" % str(time.time() - start_time))

if __name__ == '__main__':
    main()
