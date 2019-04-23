from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import sys
# import numpy as np
# import tensorflow as tf
# from datetime import datetime
#
# tf.random.set_random_seed(10)
#
# device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
# shape = (int(sys.argv[2]), int(sys.argv[2]))
# if device_name == "gpu":
#     device_name = "/gpu:0"
# else:
#     device_name = "/cpu:0"
#
# # with tf.device(device_name):
# random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
# dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
# sum_operation = tf.reduce_sum(dot_operation)
#
#
# startTime = datetime.now()
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
#         result = session.run(sum_operation)
#         print(result)
#
# # It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
# print("\n" * 5)
# print("Shape:", shape, "Device:", device_name)
# print("Time taken:", datetime.now() - startTime)
#
# print("\n" * 5)

import cv2
import time
import numpy as np
import src.face

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True
f = src.face.Face()

while True:
# while(video_capture.isOpened()):
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # time.sleep(1.0/60.0)

    if frame is None:
        break

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Recognization
    faces = f.predict1(rgb_small_frame)

    # Display the results
    for face in faces:
        name = face.name
        top = face.bounding_box[1]
        right = face.bounding_box[2]
        bottom = face.bounding_box[3]
        left = face.bounding_box[0]
        print(face.bounding_box)
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        top = int(top)
        right = int(right)
        bottom = int(bottom)
        left = int(left)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()