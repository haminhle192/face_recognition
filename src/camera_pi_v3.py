import io
import socket
import struct
import time
import threading
import picamera
import src.detection as detection

client_socket = socket.socket()
client_socket.connect(('192.168.1.212', 8000))
connection = client_socket.makefile('wb')
try:
    connection_lock = threading.Lock()
    pool = []
    pool_lock = threading.Lock()

    class ImageStreamer(threading.Thread):
        def __init__(self):
            super(ImageStreamer, self).__init__()
            print("Starting Image Streamer")
            self.detection = detection.Detection()
            self.stream = io.BytesIO()
            self.event = threading.Event()
            self.terminated = False
            self.start()

        def run(self):
            # This method runs in a background thread
            while not self.terminated:
                if self.event.wait(1):
                    try:
                        with connection_lock:
                            connection.write(struct.pack('<L', self.stream.tell()))
                            connection.flush()
                            self.stream.seek(0)
                            img = self.stream.read()
                            faces = self.detection.find_faces(img)
                            print(faces)
                            connection.write(img)
                    finally:
                        self.stream.seek(0)
                        self.stream.truncate()
                        self.event.clear()
                        with pool_lock:
                            pool.append(self)

    count = 0
    start = time.time()
    finish = time.time()

    def streams():
        global count, finish
        while finish - start < 60:
            with pool_lock:
                if len(pool)==0:
                    continue
                streamer = pool.pop()
            yield streamer.stream
            streamer.event.set()
            count += 1
            finish = time.time()

    with picamera.PiCamera() as camera:
        pool = [ImageStreamer() for i in range(2)]
        camera.resolution = (160, 160)
        # Set the framerate appropriately; too fast and we'll starve the
        # pool of streamers and crash the script
        #camera.framerate = 48
        #camera.start_preview()
        time.sleep(2)
        camera.capture_sequence(streams(), 'jpeg', use_video_port=True)
    # Shut down the streamers in an orderly fashion
    while pool:
        with pool_lock:
            streamer = pool.pop()
        streamer.terminated = True
        streamer.join()

    # Write the terminating 0-length: to the connection to let the server
    # know we're done
    with connection_lock:
        connection.write(struct.pack('<L', 0))

finally:
    connection.close()
    client_socket.close()

print('Sent %d images in %.2f seconds at %.2ffps' % (
    count, finish-start, count / (finish-start)))
