from cscore import CameraServer
#from networktables import NetworkTables

import cv2
import numpy as np

def main():

	#ip = "10.28.34.2"

	#NetworkTables.initialize(server=ip)

	cs = CameraServer.getInstance()
	cs.enableLogging()

	camera = cs.startAutomaticCapture()
	camera.setResolution(1280, 720)

	cvSink = cs.getVideo()

	outputStream = cs.putVideo("camera", 1280, 720)

	img = np.zeros(shape=(240, 320, 4), dtype=np.uint8)

	while True:
		# Tell the CvSink to grab a frame from the camera and put it
		# in the source image.  If there is an error notify the output.
		time, img = cvSink.grabFrame(img)
		if time == 0:
			# Send the output the error.
			outputStream.notifyError(cvSink.getError());
			# skip the rest of the current iteration
			continue

		#
		# Insert your image processing logic here!
		#

		# (optional) send some image back to the dashboard
		outputStream.putFrame(img)
