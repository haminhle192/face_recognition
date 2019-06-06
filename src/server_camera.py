import io
import socket
import struct
import cv2
import numpy as np
from PIL import Image

server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')
frame = 0
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
        print(open_cv_image.shape)
        cv2.imshow('Network Image',open_cv_image)
        cv2.waitKey(1)
        image.verify()
finally:
    connection.close()
    server_socket.close()