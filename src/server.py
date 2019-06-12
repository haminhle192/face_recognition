__author__ = 'vietbq'

import time
import io
import socket
import struct
import cv2
import numpy as np
from PIL import Image
import src.face as face
import json


class Server:
    def __init__(self):
        self.predictor = face.Face()
        self.server_socket = socket.socket()
        self.server_socket.bind(('0.0.0.0', 8000))
        print('Start server')
        self.server_socket.listen(0)

    def listen_client(self):
        connection = self.server_socket.accept()[0].makefile('rb')
        try:
            while True:
                image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    break
                image_stream = io.BytesIO()
                image_stream.write(connection.read(image_len))
                image_stream.seek(0)
                image = Image.open(image_stream).convert('RGB')
                open_cv_image = np.array(image)
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                self.identify_face(open_cv_image)
                image.verify()
        finally:
            connection.close()
            self.server_socket.close()

    def identify_face(self, image_data):
        start_time = time.time()
        arr_obj = self.predictor.predict1(image_data)
        data = []
        for id_obj in arr_obj:
            user = {'id': id_obj}
            data.append(user)
        duration = "Predict: %s" % str(time.time() - start_time)
        j_result = self.make_response(0, duration, data)
        print(j_result)
        return j_result

    def make_response(self, code, message, data):
        print(type(data))
        response = {'code': code, 'message': message, 'data': data}
        print(response)
        # json_result = json.dumps(response)
        return 'OK'


server = Server()
server.listen_client()