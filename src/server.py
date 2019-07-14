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
        self.server_socket = socket.socket()
        self.server_socket.bind(('0.0.0.0', 8989))
        print('Start server')
        self.predictor = face.Face()
        self.predictor.predict1(np.empty((160, 160, 3), dtype=np.uint8))
        print('Waiting for Client')
        self.server_socket.listen(0)

    def listen_client(self):
        conn, addr = self.server_socket.accept()
        read_stream = conn.makefile('rb')
        write_stream = conn.makefile('wb')
        writer = io.BytesIO()
        print('Client connected')
        try:
            while True:
                image_len = struct.unpack('<L', read_stream.read(struct.calcsize('<L')))[0]
                if image_len == 0:
                    break
                image_stream = io.BytesIO()
                image_stream.write(read_stream.read(image_len))
                image_stream.seek(0)
                img_decoded = self.decode_jpeg(image_stream.getvalue(), (160, 160, 3), dtype=np.uint8)
                # cv2.imshow('Network Image',img_decoded)
                # cv2.waitKey(1)
                # image.verify()
                response = self.identify_face(img_decoded)
                print(response)
                j_response = json.dumps(response)
                writer.seek(0)
                writer.write(j_response.encode())
                data_length = writer.tell()
                write_stream.write(struct.pack('<L', data_length))
                writer.seek(0)
                write_stream.flush()
                write_stream.write(writer.read(data_length))
                write_stream.flush()
        except Exception as e:
            print(e)
            print('Disconnect')
        finally:
            write_stream.close()
            read_stream.close()
            if self.server_socket is not None:
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