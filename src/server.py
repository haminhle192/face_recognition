__author__ = 'vietbq'

import time
import io
import socket
import struct
import cv2
import numpy as np
from PIL import Image
import src.face as face
import src.align.detect_face as detect_face
import json


class Server:
    def __init__(self):
        self.predictor = face.Face()
        self.server_socket = socket.socket()
        self.server_socket.bind(('0.0.0.0', 8989))
        print('Start server')
        self.server_socket.listen(0)

    def listen_client(self):
        connection = self.server_socket.accept()[0].makefile('rb')
        try:
            while True:
                image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    continue
                image_stream = io.BytesIO()
                image_stream.write(connection.read(image_len))
                image_stream.seek(0)
                img_decoded = self.decode_jpeg(image_stream.getvalue(), (160, 160, 3), dtype=np.uint8)
                # cv2.imshow('Network Image',img_decoded)
                # cv2.waitKey(1)
                # image.verify()
                response = self.identify_face(img_decoded)
                print(response)
                # with io.BytesIO as writer:
                #     writer.write(response.encode())
                #     data_length = writer.tell()
                #     connection.write(struct.pack('<L', data_length))
                #     writer.seek(0)
                #     connection.flush()
                #     connection.write(writer.read(data_length))
                #     print('Did send response %2f' % data_length)
        finally:
            connection.close()
            self.server_socket.close()

    def decode_jpeg(self, bytestring, shape, dtype):
        img = Image.open(io.BytesIO(bytestring))
        data = np.array(img.getdata(), dtype=dtype)
        return data.reshape(shape, order='F')

    def identify_face(self, image_data):
        start_time = time.time()
        arr_obj = self.predictor.predict1(image_data)
        data = []
        for id_obj in arr_obj:
            user = {'id': id_obj}
            data.append(user)
        duration = "Predict: %s" % str(time.time() - start_time)
        response = self.make_response(0, duration, data)
        return response

    def make_response(self, code, message, data):
        response = {'code': code, 'message': message, 'data': data}
        return response


server = Server()
server.listen_client()