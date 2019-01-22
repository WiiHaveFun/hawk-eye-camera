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
#from Stitcher import Stitcher


from threading import Thread

xFactor = 4
yFactor = 3

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
	print(mtx)
	#fx
	mtx[0,0] = mtx[0,0] / xFactor
	#cx
	mtx[0,2] = mtx[0,2] / xFactor
	#fy
	mtx[1,1] = mtx[1,1] / yFactor
	#cy
	mtx[1,2] = mtx[1,2] / yFactor
	print(mtx)	
	return mtx, dist


def warp(frame):
	undistorted_img = cv2.undistort(frame, mtx, dist, None, mtx)
	corners = [(619 / xFactor,392 / yFactor), (617 / xFactor,427 / yFactor), (663 / xFactor, 427 / yFactor),(660 / xFactor, 392 / yFactor)]
	src = np.float32([corners[0], corners[1], corners[2], corners[3]])
	dst = np.float32([[624 / xFactor,378 / yFactor],[622.75 / xFactor, 413.5 / yFactor],[657.25 / xFactor, 413.5 / yFactor],[656 / xFactor, 378 / yFactor]])

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
		print("1")
		#frameReader2 = CameraVideoStream(device_number=2).start()
		#print("2")
		#frameReader3 = CameraVideoStream(device_number=3).start()
		#print("3")
		#frameReader4 = CameraVideoStream(device_number=4).start()
		#print("4")
		frame = frameReader.read()
		warped = warp(frame)
		warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

		#frameRs = cv2.resize(frame, (1920,1080))
		warpedRs = cv2.resize(warped,(1920,1080))
		#vidBuf = np.concatenate((frameRs, warpedRs), axis=1)

		#frame2 = frameReader2.read()
		#warped2 = warp(frame2)
		#warped2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2RGB)

		#frameRs2 = cv2.resize(frame2, (1920,1080))
		#warpedRs2 = cv2.resize(warped2,(1920,1080))
		
		#stitcher = Stitcher()
		#(result, vis) = stitcher.stitch([warped, warped2], showMatches=True)
		#cv2.imshow("Keypoint Matches", vis)
		#cv2.imshow("Result", result)

		#frame3 = frameReader3.read()
		#warped3 = warp(frame3)
		#warped3 = cv2.cvtColor(warped3, cv2.COLOR_BGR2RGB)

		#frameRs = cv2.resize(frame, (1920,1080))
		#warpedRs3 = cv2.resize(warped3,(1920,1080))
		#vidBuf = np.concatenate((frameRs, warpedRs), axis=1)

		#frame4 = frameReader4.read()
		#warped4 = warp(frame4)
		#warped4 = cv2.cvtColor(warped4, cv2.COLOR_BGR2RGB)

		#frameRs2 = cv2.resize(frame2, (1920,1080))
		#warpedRs4 = cv2.resize(warped4,(1920,1080))

		#vidBuf = np.concatenate((warpedRs, warpedRs2), axis=1)
		#vidBuf2 = np.concatenate((warpedRs3, warpedRs4), axis=1)
		#vidBuf = np.concatenate((vidBuf, vidBuf2), axis=0)

		def processCam1(frameIn):
			warped = warp(frameIn)
			warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

			#frameRs=cv2.resize(frameIn, (1920,1080))
			nonlocal warpedRs
			warpedRs=cv2.resize(warped,(1920,1080))
			#nonlocal vidBuf
			#vidBuf = np.concatenate((frameRs, warpedRs), axis=1)

		#def processCam2(frameIn):
			#warped2 = warp(frameIn)
			#warped2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2RGB)

			#frameRs2=cv2.resize(frameIn, (1920,1080))
			#nonlocal warpedRs2
			#warpedRs2=cv2.resize(warped2,(1920,1080))
			#nonlocal vidBuf2
			#vidBuf2 = np.concatenate((frameRs, warpedRs), axis=1)

		#def processCam3(frameIn):
			#warped3 = warp(frameIn)
			#warped3 = cv2.cvtColor(warped3, cv2.COLOR_BGR2RGB)

			#frameRs2=cv2.resize(frameIn, (1920,1080))
			#nonlocal warpedRs3
			#warpedRs3=cv2.resize(warped3,(1920,1080))
			#nonlocal vidBuf2
			#vidBuf2 = np.concatenate((frameRs, warpedRs), axis=1)

		#def processCam4(frameIn):
			#warped4 = warp(frameIn)
			#warped4 = cv2.cvtColor(warped4, cv2.COLOR_BGR2RGB)

			#frameRs2=cv2.resize(frameIn, (1920,1080))
			#nonlocal warpedRs4
			#warpedRs4=cv2.resize(warped4,(1920,1080))
			#nonlocal vidBuf2
			#vidBuf2 = np.concatenate((frameRs, warpedRs), axis=1)

		def concatenateFrames():
			
			#nonlocal vidBuf
			#nonlocal vidBuf2
			nonlocal warpedRs#, warpedRs2#, warpedRs3, warpedRs4
			#vidBuf = np.concatenate((warpedRs, warpedRs2), axis=1)
			#vidBuf2 = np.concatenate((warpedRs3, warpedRs4), axis=1)
			#vidBuf = np.concatenate((vidBuf, vidBuf2), axis=0)


		while True:
			if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
        		# This will fail if the user closed the window; Nasties get printed to the console
				break;

			# frame = frameReader.read()
			# warped = warp(frame)
			# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

			# frameRs=cv2.resize(frame, (1920,1080))
			# warpedRs=cv2.resize(warped,(1920,1080))
			# vidBuf = np.concatenate((frameRs, warpedRs), axis=1)
			#vidBuf = processFrame(frameReader.read())
			Thread(target=processCam1(frameReader.read()))
			#Thread(target=processCam2(frameReader2.read()))
			#Thread(target=processCam3(frameReader3.read()))
			#Thread(target=processCam4(frameReader4.read()))
			
			# vidBuf = np.concatenate((warpedRs, warpedRs2), axis=1)
			# vidBuf2 = np.concatenate((warpedRs3, warpedRs4), axis=1)
			# vidBuf = np.concatenate((vidBuf, vidBuf2), axis=0)

			# frame2 = frameReader2.read()
			# warped2 = warp(frame2)
			# warped2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2RGB)

			# frameRs2=cv2.resize(frame2, (1920,1080))
			# warpedRs2=cv2.resize(warped2,(1920,1080))
			# vidBuf2 = np.concatenate((frameRs2, warpedRs2), axis=1)

			#Thread(target=concatenateFrames())

			# vidBuf = np.concatenate((vidBuf, vidBuf2), axis=0)

			Thread(target=cv2.imshow(windowName, warpedRs))

			key=cv2.waitKey(1)
			if key == 27: # Check for ESC key
				frameReader.stop()
				frameReader2.stop()
				frameReader3.stop()
				frameReader4.stop()
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
