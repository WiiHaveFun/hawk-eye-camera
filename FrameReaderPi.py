from threading import Thread
import cv2

import time

class CameraVideoStream:
    def __init__(self, device_number=1):
		# initialize the video camera stream and read the first frame
		# from the stream
        self.stream = cv2.VideoCapture(device_number)
        self.stream.set(3,640);
        self.stream.set(4,360);
        self.grabbed, self.smallFrame = self.stream.read()
		#self.frame = None
		#if self.grabbed:
		#	self.frame = cv2.resize(self.smallFrame, (0, 0), fx=2, fy=2)
		# initialize the variable used to indicate if the thread should
		# be stopped
        self.stopped = False
        self.timestamp = 0

    def start(self):
		# start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
		# keep looping infinitely until the thread is stopped
        while True:
			# if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
 
			# otherwise, read the next frame from the stream
            if(self.stream.isOpened()):
                self.grabbed, self.smallFrame = self.stream.read()
                self.timestamp = time.time()
            
			#if self.grabbed:
			#	self.frame = cv2.resize(self.smallFrame, (0, 0), fx=2, fy=2)

    def read(self):
		# return the frame most recently read
        return self.smallFrame, self.timestamp

    def isOpened(self):
        return self.stream.isOpened()
 
    def stop(self):
		# indicate that the thread should be stopped
        self.stopped = True


