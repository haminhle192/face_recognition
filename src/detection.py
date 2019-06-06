from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import src.align.detect_face as detect_face
import random
from time import sleep
import shutil
import time


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    gpu_memory_fraction = 0.3

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        print('Loading detection model ...')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, None)

    # def export_detection(
    #     self,
    #     input_dir=os.path.dirname(__file__) + "/../data/raw",
    #     output_dir=os.path.dirname(__file__) + "/../data/raw_aligned",
    #     detect_multiple_faces=False
    # ):
    #     print("Starting detect face ...")
    #     sleep(random.random())
    #     data_set = facenet.get_dataset(input_dir)
    #     output_dir = os.path.expanduser(output_dir)
    #
    #     if os.path.exists(output_dir):
    #         shutil.rmtree(output_dir)
    #
    #     os.makedirs(output_dir)
    #     print("Created output_dir.")
    #
    #     # Store some git revision info in a text file in the log directory
    #     src_path, _ = os.path.split(os.path.realpath(__file__))
    #     facenet.store_revision_info(src_path, output_dir, ' '.join([input_dir, output_dir, str(detect_multiple_faces)]))
    #
    #     # Add a random key to the filename to allow alignment using multiple processes
    #     random_key = np.random.randint(0, high=99999)
    #     bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    #
    #     with open(bounding_boxes_filename, "w") as text_file:
    #         nrof_images_total = 0
    #         nrof_successfully_aligned = 0
    #         for cls in data_set:
    #             output_class_dir = os.path.join(output_dir, cls.name)
    #             if not os.path.exists(output_class_dir):
    #                 os.makedirs(output_class_dir)
    #             for image_path in cls.image_paths:
    #                 nrof_images_total += 1
    #                 filename = os.path.splitext(os.path.split(image_path)[1])[0]
    #                 output_filename = os.path.join(output_class_dir, filename + '.png')
    #
    #                 if not os.path.exists(output_filename):
    #                     try:
    #                         img = misc.imread(image_path)
    #                     except (IOError, ValueError, IndexError) as e:
    #                         error_message = '{}: {}'.format(image_path, e)
    #                         print(error_message)
    #                     else:
    #                         if img.ndim < 2:
    #                             print('Unable to align "%s"' % image_path)
    #                             text_file.write('ERROR1 %s\n' % output_filename)
    #                             continue
    #
    #                         faces = self.find_faces(img)
    #                         nrof_faces = len(faces)
    #                         if nrof_faces == 0:
    #                             print('Unable to align "%s"' % image_path)
    #                             text_file.write('ERROR2 %s\n' % output_filename)
    #                             continue
    #                         if nrof_faces == 1:
    #                             nrof_successfully_aligned += 1
    #                             filename_base, file_extension = os.path.splitext(output_filename)
    #                             output_filename_n = "{}{}".format(filename_base, file_extension)
    #                             misc.imsave(output_filename_n, faces[0].image)
    #                             bb = faces[0].bounding_box
    #                             text_file.write(
    #                                 '%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
    #                         else:
    #                             if detect_multiple_faces:
    #                                 nrof_successfully_aligned += 1
    #                                 for i in range(nrof_faces):
    #                                     output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
    #                                     misc.imsave(output_filename_n, faces[i].image)
    #                                     bb = faces[i].bounding_box
    #                                     text_file.write(
    #                                         '%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
    #                             else:
    #                                 print('Unable to align "%s"' % image_path)
    #                                 text_file.write('ERROR3 %s\n' % output_filename)
    #
    #     print('Total number of images: %d' % nrof_images_total)
    #     print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    #     print("End detect face for training data.")

    # def find_faces_from_paths(self, image_paths):
    #     nrof_samples = len(image_paths)
    #     faces = []
    #     count_per_image = []
    #
    #     for i in xrange(nrof_samples):
    #         img = misc.imread(os.path.expanduser(image_paths[i]))
    #         img_size = np.asarray(img.shape)[0:2]
    #         bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet,
    #                                                     self.threshold, self.factor)
    #         count_per_image.append(len(bounding_boxes))
    #
    #         for bb in bounding_boxes:
    #             face = Face()
    #             face.container_image = img
    #             face.bounding_box = np.zeros(4, dtype=np.int32)
    #
    #             face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
    #             face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
    #             face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
    #             face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
    #             cropped = img[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
    #             face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
    #
    #             faces.append(face)
    #
    #     return faces, count_per_image, nrof_samples

    def find_faces(self, image):
        start = time.time()
        image = image[:, :, 0:3]

        faces = []

        bounding_boxes, _ = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet,
                                                    self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = cropped
            # face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        print(time.time() - start)

        return faces


