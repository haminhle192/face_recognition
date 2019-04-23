from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import math
import src.facenet as facenet


class Encoder:
    def __init__(
            self,
            sess=tf.Session(),
            pre_trained_model=os.path.dirname(__file__) + "/../pre_trained_models/CASIA.pb",
            batch_size=32,
            image_size=160
    ):
        self.sess = sess
        self.pre_trained_model = pre_trained_model
        self.batch_size = batch_size
        self.image_size = image_size
        print("Loading pre trained model ...")
        with self.sess.as_default():
            facenet.load_model(self.pre_trained_model)

    def generate_embeddings(self, image_paths):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # print('Calculating embeddings ...')
        nrof_images = len(image_paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            # print(i+1, '/', nrof_batches_per_epoch)
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            paths_batch = image_paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, self.image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = self.sess.run(embeddings, feed_dict=feed_dict)
        return emb_array

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

