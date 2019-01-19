import numpy as np
import cv2
import glob
import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_uint
from skimage import io
from pathlib import Path

import pickle

import sys
import argparse
from FrameReader import CameraVideoStream

from threading import Thread

def init():
	distortion_correction_file = Path("./distortion_correction_pickle.p")
	# check if we already created the calibration file with coefficients
	if distortion_correction_file.is_file():
		# load the coefficients to undistort the camera image
		with open('./distortion_correction_pickle.p', mode='rb') as f:
			calibration_file = pickle.load(f)
			mtx, dist = calibration_file['mtx'], calibration_file['dist']
	else:
		print('Calibration does not exist. Please run the cell above to create it first.')
	return mtx, dist


def warp(frame):
	undistorted_img = cv2.undistort(frame, mtx, dist, None, mtx)
	corners = [(619,392), (617,427), (663, 427),(660, 392)]
	src = np.float32([corners[0], corners[1], corners[2], corners[3]])
	dst = np.float32([[624,378],[622.75, 413.5],[657.25, 413.5],[656, 378]])

	def perspective_warp(img):
    	# Grab the image shape
		img_size = (img.shape[1], img.shape[0])

        # Given src and dst points, calculate the perspective transform matrix
		M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
		warped = cv2.warpPerspective(img, M, img_size)

        # Return the resulting image and matrix
		return warped

	warped_image = perspective_warp(undistorted_img)
	warped_cv2_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR)

	return warped_cv2_image

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device", dest="video_device",
                        help="Video device # of USB webcam (/dev/video?) [0]",
                        default=0, type=int)
    arguments = parser.parse_args()
    return arguments

def displayFrames():
	if True:
		
		windowName = "Hawk Eye"
		cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(windowName,1280,720)
		cv2.moveWindow(windowName,0,0)
		cv2.setWindowTitle(windowName,"Hawk Eye Vision")
		frameReader = CameraVideoStream(device_number=1).start()
		frameReader2 = CameraVideoStream(device_number=2).start()

		while True:
			if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
        	# This will fail if the user closed the window; Nasties get printed to the console
				break;

			frame = frameReader.read()
			warped = warp(frame)
			warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

			frameRs=cv2.resize(frame, (1920,1080))
			warpedRs=cv2.resize(warped,(1920,1080))
			vidBuf = np.concatenate((frameRs, warpedRs), axis=1)

			frame2 = frameReader2.read()
			warped2 = warp(frame2)
			warped2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2RGB)

			frameRs2=cv2.resize(frame2, (1920,1080))
			warpedRs2=cv2.resize(warped2,(1920,1080))
			vidBuf2 = np.concatenate((frameRs2, warpedRs2), axis=1)

			vidBuf = np.concatenate((vidBuf, vidBuf2), axis=0)

			Thread(target=cv2.imshow(windowName, vidBuf))

			key=cv2.waitKey(1)
			if key == 27: # Check for ESC key
				frameReader.stop()
				frameReader2.stop()
				cv2.destroyAllWindows()
				break ;

	else:
		print ("camera open failed")

if __name__ == '__main__':
	arguments = parse_cli_args()
	print("Called with args:")
	print(arguments)
	print("OpenCV version: {}".format(cv2.__version__))
	print("Device Number:",arguments.video_device)
	mtx, dist = init()
	displayFrames()
	cv2.destroyAllWindows()
