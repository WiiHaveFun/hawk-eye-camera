import cv2
import numpy as np
from pathlib import Path

import pickle

import math

xFactor = 1
yFactor = 1

def init():
	distortion_correction_file = Path("./distortion_correction_pickle_regular_camera.p")
	# check if we already created the calibration file with coefficients
	if distortion_correction_file.is_file():
		# load the coefficients to undistort the camera image
		with open('./distortion_correction_pickle_regular_camera.p', mode='rb') as f:
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

def getPos():
	img1 = cv2.imread('./HomographyFrame1.jpg')
	img2 = cv2.imread('./HomographyFrame2.jpg')

	dst = cv2.undistort(img1, mtx, dist, None, mtx)
	cv2.imshow("undistort", dst)

	#top four corners

	corners1 = np.array([[684,459], [763,482], [1087,483], [1167,468]], dtype=np.float64)
	corners2 = np.array([[538,482], [565,489], [662,494], [684,485]], dtype=np.float64)

	# outer most corners
	corners1 = np.array([[618,698], [684,458], [1163,465], [1226,701]], dtype=np.float64)
	corners2 = np.array([[516,585], [538,480], [684,484], [698,574]], dtype=np.float64)
	
	#corners1 = np.array([[459,684], [482,763], [483,1087], [468,1167]])
	#corners2 = np.array([[482,538], [489,565], [494,662], [485,684]])

	H, _ = cv2.findHomography(corners2, corners1)

	print("H", H)

	#img1_size = (img1.shape[1], img1.shape[0])

	#print(img1.shape[:2])

	dst = cv2.warpPerspective(img1, H, (1920,1080))
	
	cv2.imshow("dst", dst)

	# in inches
	# top corners
	#objectPoints = np.array([[-5.936,0,0], [-4,-0.5,0], [4,-0.5,0], [5.936,0,0]])

	# outer most corners
	objectPoints = np.array([[-7.316,-5.324,0], [-5.936,0,0], [5.936,0,0], [7.316,-5.324,0]])

	retval, rvec1, tvec1 = cv2.solvePnP(objectPoints, corners1, mtx, dist, cv2.SOLVEPNP_P3P)
	print(retval)

	print(rvec1)
	print(tvec1)

	retval, rvec2, tvec2 = cv2.solvePnP(objectPoints, corners2, mtx, dist, cv2.SOLVEPNP_P3P)
	print(retval)

	print(rvec2)
	print(tvec2)
	
	cv2.waitKey(0)

	def computeC2MC1(R1, tvec1, R2, tvec2):
		
		print("R1", R1)
		print("R1 rotated", np.transpose(R1))
		print("R2", R2)

		R_1to2 = np.matmul(R2, np.transpose(R1))

		print("R1to2",R_1to2)

		R_1to2, _ = cv2.Rodrigues(R_1to2)
		
		#print(R_1to2[1][0])
		#R_1to2[0][0] = math.degrees(R_1to2[0][0])
		#R_1to2[0][1] = math.degrees(R_1to2[0][1])
		#R_1to2[0][2] = math.degrees(R_1to2[0][2])

		print(R_1to2[0])
		R_1to2[0] = math.degrees(R_1to2[0])
		R_1to2[1] = math.degrees(R_1to2[1])
		R_1to2[2] = math.degrees(R_1to2[2])

		print("r1 rotated * t1", np.matmul(-np.transpose(R1), tvec1))

		temp = np.matmul(-np.transpose(R1), tvec1)

		tvec_1to2 = np.matmul(R2, temp) + tvec2

		print("tvec2", tvec2)
		print("expected a scalar", R2 * (-np.transpose(R1)*tvec1))
		print(type(R2))

		#tvec_1to2 = cv2.Rodrigues(tvec_1to2)

		return R_1to2, tvec_1to2

	R1to2, tvec1to2 = computeC2MC1(rvec1, tvec1, rvec2, tvec2)
	
	print("between point 1 and 2")
	print(R1to2)
	print(tvec1to2)

if __name__ == '__main__':
	mtx, dist = init()
	getPos()
