#!/usr/bin/env python3
# saves a frame to the pi every five seconds to use for calibration

# import the necessary packages
import numpy as np
import cv2
import time

import cscore as cs
import logging

from networktables import NetworkTables

## init camera server and network tables
logging.basicConfig(level=logging.DEBUG)
ip = "10.28.34.2"
NetworkTables.initialize(server=ip)
sd = NetworkTables.getTable("SmartDashboard")

cs = cs.CameraServer.getInstance()
outputStream = cs.putVideo("cal frame", 640, 360)

cap = cv2.VideoCapture("/dev/video1")
# 720p calibration images
cap.set(3,1280)
cap.set(4,720)

# Set Exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, 0.05)

i = 1
ret, frame = cap.read()
while(True):
    startTime = time.time()
    while(time.time() - startTime < 5):
        ret, frame = cap.read()
        img = cv2.resize(frame, (640,360))
        outputStream.putFrame(img)
#    if i < 21:
#        cv2.imwrite('calibration' + str(i) + '.jpg', frame)
#        print("saved ", i)
#        i += 1
