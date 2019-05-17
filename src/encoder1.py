from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import math
import src.facenet as facenet
import time


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
            print(i+1, '/', nrof_batches_per_epoch)
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, nrof_images)
            paths_batch = image_paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, self.image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = self.sess.run(embeddings, feed_dict=feed_dict)
        return emb_array

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
        embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        # st = time.time()
        result = self.sess.run(embeddings, feed_dict=feed_dict)[0]
        # print("emb time: %s" % str(time.time() - st))
        return result

    def export_embeddings(self, data_dir, embeddings_name, labels_name, label_strings_name):
        train_set = facenet.get_dataset(data_dir)
        image_paths, labels = facenet.get_image_paths_and_labels(train_set)
        # fetch the classes (labels as strings) exactly as it's done in get_dataset
        path_exp = os.path.expanduser(data_dir)
        classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
        classes.sort()
        # get the label strings
        label_strings = [name for name in classes if os.path.isdir(os.path.join(path_exp, name))]

        emb_array = self.generate_embeddings(image_paths)
        labels = np.array(labels)
        label_strings = np.array(label_strings)
        np.save(embeddings_name, emb_array)
        np.save(labels_name, labels)
        np.save(label_strings_name, label_strings[labels])
