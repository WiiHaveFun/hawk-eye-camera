#!/usr/bin/env python3
# import the necessary packages

import cv2
import numpy as np

from networktables import NetworkTables
import cscore as cs
import logging

from FrameReaderPi import CameraVideoStream

from threading import Thread

import time

counter = 0

disconnectedFrame = np.zeros((360,640,3), np.uint8)
cv2.putText(disconnectedFrame, "The Lifecam is disconnected! Please check Raspberry Pi.", (70, 210), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 255, 255))
disconnectedFrameWithTriangle = np.zeros((360,640,3), np.uint8)
cv2.putText(disconnectedFrameWithTriangle, "The Lifecam is disconnected! Please check Raspberry Pi.", (70, 210), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 255, 255))
cv2.line(disconnectedFrameWithTriangle, (270,130), (370,130), (0,255,255), 5)
cv2.line(disconnectedFrameWithTriangle, (270,130), (320,30), (0,255,255), 5)
cv2.line(disconnectedFrameWithTriangle, (370,130), (320,30), (0,255,255), 5)

cv2.line(disconnectedFrameWithTriangle, (320,95), (320,60), (0,255,255), 5)
cv2.line(disconnectedFrameWithTriangle, (320,110), (320,110), (0,255,255), 10)


# Calculate Homography
# Both are image number 2 in calibration set
# Source Points (Lifecam)
srcCorners = np.float32([[340,311], [331,615], [767,633], [780,324]])
# Destination Points (Picam)
dstCorners = np.float32([[371,336], [367,583], [726,591], [730,346]])
# Calculate Homography from Lifecam to Picam
H, _ = cv2.findHomography(srcCorners, dstCorners)
#print("ret", ret)
print("homography", H)

# For bounding box
transformedBoxP1x = 0
transformedBoxP1y = 0
transformedBoxP2x = 0
transformedBoxP2y = 0
transformedBoxP3x = 0
transformedBoxP3y = 0
transformedBoxP4x = 0
transformedBoxP4y = 0

# init camera server and network tables
logging.basicConfig(level=logging.DEBUG)
ip = "10.28.34.2"
NetworkTables.initialize(server=ip)
sd = NetworkTables.getTable("SmartDashboard")

cs = cs.CameraServer.getInstance()

outputStream = cs.putVideo("driver view", 640, 360)

def transformBoundingBox():
	transformedBoxP1x = sd.getNumber("transformedBoxP1x", "0")
	transformedBoxP1y = sd.getNumber("transformedBoxP1y", "0")

	transformedBoxP2x = sd.getNumber("transformedBoxP2x", "0")
	transformedBoxP2y = sd.getNumber("transformedBoxP2y", "0")

	transformedBoxP3x = sd.getNumber("transformedBoxP3x", "0")
	transformedBoxP3y = sd.getNumber("transformedBoxP3y", "0")

	transformedBoxP4x = sd.getNumber("transformedBoxP4x", "0")
	transformedBoxP4y = sd.getNumber("transformedBoxP4y", "0")

	return np.array([[[transformedBoxP1x,transformedBoxP1y], [transformedBoxP2x,transformedBoxP2y], [transformedBoxP3x,transformedBoxP3y], [transformedBoxP4x,transformedBoxP4y]]], np.float32)

	print("bounding",boundingBox)
	print(H)
	cv2.perspectiveTransform(boundingBox, H, boundingBox)
	print("bounding",boundingBox)

cap = CameraVideoStream("/dev/video0").start()

while(True):

	if cap.isOpened():
		frame, frameAquiredTime = cap.read()
		#print("bounding",boundingBox)
		boundingBox = transformBoundingBox()
		cv2.perspectiveTransform(boundingBox, H, boundingBox)
		#print("bounding",boundingBox)
		if sd.getBoolean("Target Status", False):
			if ((boundingBox[0][0][0] != boundingBox[0][1][0]) and (boundingBox[0][0][1] != boundingBox[0][1][1])) and ((boundingBox[0][2][0] != boundingBox[0][3][0]) and (boundingBox[0][2][1] != boundingBox[0][3][1])):
			    cv2.line(frame, (boundingBox[0][0][0],boundingBox[0][0][1]), (boundingBox[0][1][0],boundingBox[0][1][1]), (0,255,0), 5)
			    cv2.line(frame, (boundingBox[0][1][0],boundingBox[0][1][1]), (boundingBox[0][2][0],boundingBox[0][2][1]), (0,255,0), 5)
			    cv2.line(frame, (boundingBox[0][2][0],boundingBox[0][2][1]), (boundingBox[0][3][0],boundingBox[0][3][1]), (0,255,0), 5)
			    cv2.line(frame, (boundingBox[0][3][0],boundingBox[0][3][1]), (boundingBox[0][0][0],boundingBox[0][0][1]), (0,255,0), 5)

	
		outputStream.putFrame(frame)
	else:
		if counter % 2 == 0:
			outputStream.putFrame(disconnectedFrame)
			time.sleep(1)
		elif counter % 2 == 1:
			outputStream.putFrame(disconnectedFrameWithTriangle)
			time.sleep(1)
		counter+=1