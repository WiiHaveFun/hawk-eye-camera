#!/usr/bin/env python3
#
# This is a NetworkTables client (eg, the DriverStation/coprocessor side).
# You need to tell it the IP address of the NetworkTables server (the
# robot or simulator).
#
# When running, this will continue incrementing the value 'dsTime', and the
# value should be visible to other networktables clients and the robot.
#

import sys
import time
import cv2
from networktables import NetworkTables
import numpy as np
import cscore as cs

# To see messages from networktables, you must setup logging
import logging

logging.basicConfig(level=logging.DEBUG)

camera = cs.CvSource("cvsource", cs.VideoMode.PixelFormat.kMJPEG, 320, 240, 30)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

#if len(sys.argv) != 2:
#    print("Error: specify an IP to connect to!")
#    exit(0)

#ip = sys.argv[1]
ip = "10.28.34.2"

NetworkTables.initialize(server=ip)

sd = NetworkTables.getTable("SmartDashboard")

img = cv2.imread('transparent.jpg',-1)
img = np.asarray(img)
img = None
i = 0
while True:
    print("robotTime:", sd.getNumber("robotTime", "N/A"))

    sd.putNumber("dsTime", i)
    #sd.putVideo("frame", img)
    time.sleep(1)
    i += 1
    retval, img = cap.read(img)
    if retval:
        camera.putFrame(img)
    mjpegServer = cs.MjpegServer("httpserver", 8081)
    mjpegServer.setSource(camera)

