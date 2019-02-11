#!/usr/bin/env python3
# import the necessary packages
import datetime
import time
import numpy as np
import cv2
import math

from pathlib import Path
import pickle

from networktables import NetworkTables
import cscore as cs
import logging

from FrameReaderPi import CameraVideoStream

from threading import Thread

startTime = time.time()

tvecXSaved = np.array([0])
tvecYSaved = np.array([0])
tvecZSaved = np.array([0])

# Calculate Homography
# Both are image number 2
# Source Points (Lifecam)
srcCorners = np.float32([[340,311], [331,615], [767,633], [780,324]])
# Destination Points (Picam)
dstCorners = np.float32([[371,336], [367,583], [726,591], [730,346]])
# Calculate Homography from Lifecam to Picam
H, _ = cv2.findHomography(srcCorners, dstCorners)
#print("ret", ret)
print("homography", H)

# Cal for pi cam is 720p

mtx = None
dist = None
# 1280 by 720
#xFactor = 1.5
#yFactor = 1.5
# 640 by 360
#xFactor = 3
#yFactor = 3
# 480 by 270
#xFactor = 4
#yFactor = 4

# 1280 by 720
#xFactor = 1
#yFactor = 1
# 640 by 360
xFactor = 2
yFactor = 2

# Transformation for overlays
xTrans = 22
yTrans = -50
sf = 0.9
spacing = -10

def findCentroid(x1, x2, x3, x4, y1, y2, y3, y4):
    x = (x1 + x2 + x3 + x4) / 4
    y = (y1 + y2 + y3 + y4) / 4
    return x, y

def transformAndScale(cx, cy, x, y, tx, ty, scale):
    x = scale * (x - cx) + cx
    y = scale * (y - cy) + cy
    x = int(x + tx)
    y = int(y + ty)
    return x, y

transformedP1x = 0
transformedP1y = 0
transformedP2x = 0
transformedP2y = 0
transformedP3x = 0
transformedP3y = 0
transformedP4x = 0
transformedP4y = 0

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
outputStream = cs.putVideo("vision", 640, 360)
outputStream2 = cs.putVideo("test", 640, 360)

# calibrated for 1080p
distortion_correction_file = Path("./distortion_correction_pickle_picam720.p")
# check if we already created the calibration file with coefficients
if distortion_correction_file.is_file():
	# load the coefficients to undistort the camera image
	with open('./distortion_correction_pickle_picam720.p', mode='rb') as f:
		calibration_file = pickle.load(f)
		mtx, dist = calibration_file['mtx'], calibration_file['dist']
else:
	print('Calibration does not exist. Please run the cell above to create it first.')
print("mtx: ", mtx)
print("dist: ", dist)
#fx
mtx[0,0] = mtx[0,0] / xFactor
#cx
mtx[0,2] = mtx[0,2] / xFactor
#fy
mtx[1,1] = mtx[1,1] / yFactor
#cy
mtx[1,2] = mtx[1,2] / yFactor
print(mtx)

objectPoints = np.array([[-7.316,-5.324,0], [-5.936,0,0], [5.936,0,0], [7.316,-5.324,0], [-4,-0.5,0], [4,-0.5,0]])
trihedron = np.array([[0,-2.754,0],[2,-2.754,0],[0,-0.754,0],[0,-2.754,2]])

#Class to examine Frames per second of camera stream. Currently not used.
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()

#image_width = 1920
#image_height = 1080
#image_width = 1280
#image_height = 720
image_width = 640
image_height = 360
#image_width = 480
#image_height = 270

diagonalView = math.radians(105)

#16:9 aspect ratio
horizontalAspect = 16
verticalAspect = 9

#Reasons for using diagonal aspect is to calculate horizontal field of view.
diagonalAspect = math.hypot(horizontalAspect, verticalAspect)
#Calculations: http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
horizontalView = math.atan(math.tan(diagonalView/2) * (horizontalAspect / diagonalAspect)) * 2
verticalView = math.atan(math.tan(diagonalView/2) * (verticalAspect / diagonalAspect)) * 2

H_FOCAL_LENGTH = image_width / (2*math.tan((horizontalView/2)))
V_FOCAL_LENGTH = image_height / (2*math.tan((verticalView/2)))

# Checks if tape contours are worthy based off of contour area and (not currently) hull area
def checkContours(cntSize, hullSize):
    return cntSize > (image_width / 6)


def getEllipseRotation(image, cnt):
    try:
        # Gets rotated bounding ellipse of contour
        ellipse = cv2.fitEllipse(cnt)
        centerE = ellipse[0]
        # Gets rotation of ellipse; same as rotation of contour
        rotation = ellipse[2]
        # Gets width and height of rotated ellipse
        widthE = ellipse[1][0]
        heightE = ellipse[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, widthE, heightE)

        #cv2.ellipse(image, ellipse, (23, 184, 80), 3)
        return rotation
    except:
        # Gets rotated bounding rectangle of contour
        rect = cv2.minAreaRect(cnt)
        # Creates box around that rectangle
        box = cv2.boxPoints(rect)
        # Not exactly sure
        box = np.int0(box)
        # Gets center of rotated rectangle
        center = rect[0]
        # Gets rotation of rectangle; same as rotation of contour
        rotation = rect[2]
        # Gets width and height of rotated rectangle
        width = rect[1][0]
        height = rect[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, width, height)
        return rotation

#Forgot how exactly it works, but it works!
def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)

def calculateYaw(pixelX, centerX, hFocalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
    return round(yaw)

# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculatePitch(pixelY, centerY, vFocalLength):
    pitch = math.degrees(math.atan((pixelY - centerY) / vFocalLength))
    # Just stopped working have to do this:
    pitch *= -1
    return round(pitch)

# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findTape(contours, image, centerX, centerY, image2):
    screenHeight, screenWidth, channels = image.shape;
    #Seen vision targets (correct angle, adjacent to each other)
    targets = []

    if len(contours) >= 2:
        #Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        biggestCnts = []
        for cnt in cntsSorted:
            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Get convex hull (bounding polygon on contour)
            hull = cv2.convexHull(cnt)
            # Calculate Contour area
            cntArea = cv2.contourArea(cnt)
            # calculate area of convex hull
            hullArea = cv2.contourArea(hull)
            # Filters contours based off of size
            if (checkContours(cntArea, hullArea)):
                ### MOSTLY DRAWING CODE, BUT CALCULATES IMPORTANT INFO ###
                # Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                if(len(biggestCnts) < 13):
                    #### CALCULATES ROTATION OF CONTOUR BY FITTING ELLIPSE ##########
                    rotation = getEllipseRotation(image, cnt)

                    # Calculates yaw of contour (horizontal position in degrees)
                    yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    # Calculates yaw of contour (horizontal position in degrees)
                    pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)

                    ##### DRAWS CONTOUR######
                    # Gets rotated bounding rectangle of contour
                    rect = cv2.minAreaRect(cnt)
                    # Creates box around that rectangle
                    box = cv2.boxPoints(rect)
                    #print("box ", box)
                    # Not exactly sure
                    box = np.int0(box)
                    # Draws rotated rectangle
                    #cv2.drawContours(image, [box], 0, (23, 184, 80), 3)


                    # Calculates yaw of contour (horizontal position in degrees)
                    yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    # Calculates yaw of contour (horizontal position in degrees)
                    pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)


                    # Draws a vertical white line passing through center of contour
                    cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                    # Draws a white circle at center of contour
                    cv2.circle(image, (cx, cy), 6, (255, 255, 255))

                    # Draws the contours
                    cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                    # Gets the (x, y) and radius of the enclosing circle of contour
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    # Rounds center of enclosing circle
                    center = (int(x), int(y))
                    # Rounds radius of enclosning circle
                    radius = int(radius)
                    # Makes bounding rectangle of contour
                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    boundingRect = cv2.boundingRect(cnt)
                    # Draws countour of bounding rectangle and enclosing circle in green
                    #cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)

                    #cv2.circle(image, center, radius, (23, 184, 80), 1)
                    rawHull = cv2.convexHull(cnt, clockwise=True)
                    rawHullPeri = cv2.arcLength(rawHull, closed=True)
                    #approx = cv2.approxPolyDP(cnt,0.052*cv2.arcLength(cnt,True),True)
                    approx = cv2.approxPolyDP(rawHull,0.045*rawHullPeri,True)
                    #print len(approx)
                    if len(approx)==4:
                        #print ("target detected")
                        #print(approx)
                        p1 = approx[0][0]
                        p2 = approx[1][0]
                        p3 = approx[2][0]
                        p4 = approx[3][0]
                        cv2.line(blur, (p1[0],p1[1]), (p2[0],p2[1]), (255,0,0), 2)
                        cv2.line(blur, (p2[0],p2[1]), (p3[0],p3[1]), (255,0,0), 2)
                        cv2.line(blur, (p3[0],p3[1]), (p4[0],p4[1]), (255,0,0), 2)
                        cv2.line(blur, (p4[0],p4[1]), (p1[0],p1[1]), (255,0,0), 2)
                        centroidX, centroidY = findCentroid(p1[0],p2[0],p3[0],p4[0], p1[1], p2[1], p3[1], p4[1])
                        
                        transformedP1x, transformedP1y = transformAndScale(centroidX, centroidY, p1[0], p1[1], xTrans, yTrans, sf)
                        transformedP2x, transformedP2y = transformAndScale(centroidX, centroidY, p2[0], p2[1], xTrans, yTrans, sf)
                        transformedP3x, transformedP3y = transformAndScale(centroidX, centroidY, p3[0], p3[1], xTrans, yTrans, sf)
                        transformedP4x, transformedP4y = transformAndScale(centroidX, centroidY, p4[0], p4[1], xTrans, yTrans, sf)
                        
                        cv2.line(frame2, (transformedP1x,transformedP1y), (transformedP2x,transformedP2y), (255,0,0), 2)
                        cv2.line(frame2, (transformedP2x,transformedP2y), (transformedP3x,transformedP3y), (255,0,0), 2)
                        cv2.line(frame2, (transformedP3x,transformedP3y), (transformedP4x,transformedP4y), (255,0,0), 2)
                        cv2.line(frame2, (transformedP4x,transformedP4y), (transformedP1x,transformedP1y), (255,0,0), 2)

                    # Appends important info to array
                    if [cx, cy, rotation, cnt] not in biggestCnts:
                         biggestCnts.append([cx, cy, rotation, cnt, box, approx])


        # Sorts array based on coordinates (leftmost to rightmost) to make sure contours are adjacent
        biggestCnts = sorted(biggestCnts, key=lambda x: x[0])
        # Target Checking
        for i in range(len(biggestCnts) - 1):
            #Rotation of two adjacent contours
            tilt1 = biggestCnts[i][2]
            tilt2 = biggestCnts[i + 1][2]

            #x coords of contours
            cx1 = biggestCnts[i][0]
            cx2 = biggestCnts[i + 1][0]

            cy1 = biggestCnts[i][1]
            cy2 = biggestCnts[i + 1][1]

            #box of contours
            box1 = biggestCnts[i][4]
            box2 = biggestCnts[i + 1][4]

            #poly of contours
            target1 = biggestCnts[i][5]
            target2 = biggestCnts[i + 1][5]
            # If contour angles are opposite
            if (np.sign(tilt1) != np.sign(tilt2)):
                centerOfTarget = math.floor((cx1 + cx2) / 2)
                #ellipse negative tilt means rotated to right
                #Note: if using rotated rect (min area rectangle)
                #      negative tilt means rotated to left
                # If left contour rotation is tilted to the left then skip iteration
                if (tilt1 > 0):
                    if (cx1 < cx2):
                        continue
                # If left contour rotation is tilted to the left then skip iteration
                if (tilt2 > 0):
                    if (cx2 < cx1):
                        continue
                #Angle from center of camera to target (what you should pass into gyro)
                yawToTarget = calculateYaw(centerOfTarget, centerX, H_FOCAL_LENGTH)
                #Make sure no duplicates, then append
                if [centerOfTarget, yawToTarget] not in targets:
                    targets.append([centerOfTarget, yawToTarget, box1, box2, target1, target2])
    #Check if there are targets seen
    if (len(targets) > 0):
        # pushes that it sees vision target to network tables
        print("tapeDetected: ", True)
        #Sorts targets based on x coords to break any angle tie
        targets.sort(key=lambda x: math.fabs(x[0]))
        finalTarget = min(targets, key=lambda x: math.fabs(x[1]))
        # Puts the yaw on screen
        #Draws yaw of target + line where center of target is
#        cv2.putText(image, "Yaw: " + str(finalTarget[1]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
#                    (255, 255, 255))
        cv2.line(image, (finalTarget[0], screenHeight), (finalTarget[0], 0), (255, 0, 0), 2)
        cv2.line(image2, (finalTarget[0] + xTrans, screenHeight), (finalTarget[0]  + xTrans, 0), (255, 0, 0), 2)
        
        currentAngleError = finalTarget[1]
        finalBox1 = finalTarget[2]
        finalBox2 = finalTarget[3]

        # a more accurate outline of the target, sorted from smallest to greatest x value (left to right)
        finalPoly1 = finalTarget[4]
        finalPoly2 = finalTarget[5]
        finalPoly1_SortedY = sorted(finalPoly1, key=lambda k: [k[0][1]])
        finalPoly1Top = finalPoly1_SortedY[0]
        finalPoly1Inside = finalPoly1_SortedY[1]
        # avoids any corner errors
        if finalPoly1Top[0][0] > finalPoly1Inside[0][0]:
            temp = finalPoly1Top
            finalPoly1Top = finalPoly1Inside
            finalPoly1Inside = temp
        print(finalPoly1Inside)
        finalPoly2_SortedY = sorted(finalPoly2, key=lambda k: [k[0][1]])
        finalPoly2Top = finalPoly2_SortedY[0]
        finalPoly2Inside = finalPoly2_SortedY[1]
        if finalPoly2Top[0][0] < finalPoly2Inside[0][0]:
            temp = finalPoly2Top
            finalPoly2Top = finalPoly2Inside
            finalPoly2Inside = temp
        print(finalPoly2Inside)
        
        finalPoly1_SortedX = sorted(finalPoly1, key=lambda k: [k[0][0]])
        finalPoly1Outside = finalPoly1_SortedX[0]
        finalPoly2_SortedX = sorted(finalPoly2, key=lambda k: [k[0][0]], reverse=True)
        finalPoly2Outside = finalPoly2_SortedX[0]
        #print("unsorted", finalPoly1)
        #print("sorted", finalPoly1_Sorted)
        #print("unsorted 2", finalPoly2)
        #print("sorted 2", finalPoly2_Sorted)
        #corners = np.array([finalBox1[1], finalBox1[2], finalBox2[2], finalBox2[3]], dtype=np.float32)
        corners = np.array([finalPoly1Outside, finalPoly1Top, finalPoly2Top, finalPoly2Outside, finalPoly1Inside, finalPoly2Inside], dtype=np.float32)
        #corners = np.array([[462,189], [478,80], [616,91], [633,166]], dtype=np.float32)
        #corners = np.array([[462,190], [478,80], [616,91], [633,167]], dtype=np.float32)
        # pushes vision target angle to network tables
        #print("tapeYaw: ", currentAngleError)
        #print("box 1: ", finalBox1)
        #print("box 2: ", finalBox2)
        print("corners: ", corners)
        retval, rvec, tvec = cv2.solvePnP(objectPoints, corners, mtx, dist, cv2.SOLVEPNP_P3P)
        #points for trihedron
        trihedronPoints, _ = cv2.projectPoints(trihedron, rvec, tvec, mtx, dist)
        #draw trihedron
        #prevents out-of-bounds integers
        if ((trihedronPoints[0][0][0] <= 10000) and (trihedronPoints[0][0][0] >= -10000)) and ((trihedronPoints[0][0][1] <= 10000) and (trihedronPoints[0][0][1] >= -10000)) and ((trihedronPoints[1][0][0] <= 10000) and (trihedronPoints[1][0][0] >= -10000)) and ((trihedronPoints[1][0][1] <= 10000) and (trihedronPoints[1][0][1] >= -10000)) and ((trihedronPoints[2][0][0] <= 10000) and (trihedronPoints[2][0][0] >= -10000)) and ((trihedronPoints[2][0][1] <= 10000) and (trihedronPoints[2][0][1] >= -10000)) and ((trihedronPoints[3][0][0] <= 10000) and (trihedronPoints[3][0][0] >= -10000)) and ((trihedronPoints[3][0][1] <= 10000) and (trihedronPoints[3][0][1] >= -10000)):
            cv2.line(image, (int(trihedronPoints[0][0][0]), int(trihedronPoints[0][0][1])), (int(trihedronPoints[1][0][0]), int(trihedronPoints[1][0][1])), (255,0,0), 3)
            cv2.line(image, (int(trihedronPoints[0][0][0]), int(trihedronPoints[0][0][1])), (int(trihedronPoints[2][0][0]), int(trihedronPoints[2][0][1])), (0,255,0), 3)
            cv2.line(image, (int(trihedronPoints[0][0][0]), int(trihedronPoints[0][0][1])), (int(trihedronPoints[3][0][0]), int(trihedronPoints[3][0][1])), (0,0,255), 3)
            cv2.line(image2, (int(trihedronPoints[0][0][0] + xTrans), int(trihedronPoints[0][0][1]) + yTrans), (int(trihedronPoints[1][0][0] + xTrans), int(trihedronPoints[1][0][1]) + yTrans), (255,0,0), 3)
            cv2.line(image2, (int(trihedronPoints[0][0][0] + xTrans), int(trihedronPoints[0][0][1]) + yTrans), (int(trihedronPoints[2][0][0] + xTrans), int(trihedronPoints[2][0][1]) + yTrans), (0,255,0), 3)
            cv2.line(image2, (int(trihedronPoints[0][0][0] + xTrans), int(trihedronPoints[0][0][1]) + yTrans), (int(trihedronPoints[3][0][0] + xTrans), int(trihedronPoints[3][0][1]) + yTrans), (0,0,255), 3)
        
        cv2.line(blur, (corners[0][0][0],corners[0][0][1]), (corners[1][0][0],corners[1][0][1]), (0,0,255), 5)
        cv2.line(blur, (corners[1][0][0],corners[1][0][1]), (corners[2][0][0],corners[2][0][1]), (0,0,255), 5)
        cv2.line(blur, (corners[2][0][0],corners[2][0][1]), (corners[3][0][0],corners[3][0][1]), (0,0,255), 5)
        cv2.line(blur, (corners[3][0][0],corners[3][0][1]), (corners[0][0][0],corners[0][0][1]), (0,0,255), 5)

        boxCentroidX, boxCentroidY = findCentroid(corners[0][0][0],corners[1][0][0],corners[2][0][0],corners[3][0][0], corners[0][0][1],corners[1][0][1],corners[2][0][1],corners[3][0][1])

        transformedBoxP1x, transformedBoxP1y = transformAndScale(boxCentroidX, boxCentroidY, corners[0][0][0], corners[0][0][1], xTrans, yTrans, sf)
        transformedBoxP2x, transformedBoxP2y = transformAndScale(boxCentroidX, boxCentroidY, corners[1][0][0], corners[1][0][1], xTrans, yTrans, sf)
        transformedBoxP3x, transformedBoxP3y = transformAndScale(boxCentroidX, boxCentroidY, corners[2][0][0], corners[2][0][1], xTrans, yTrans, sf)
        transformedBoxP4x, transformedBoxP4y = transformAndScale(boxCentroidX, boxCentroidY, corners[3][0][0], corners[3][0][1], xTrans, yTrans, sf)

#        boundingBox = np.zeros((360,360,3), np.uint8)
#
#        cv2.line(boundingBox, (transformedBoxP1x,transformedBoxP1y), (transformedBoxP2x,transformedBoxP2y), (0,0,255), 5)
#        cv2.line(boundingBox, (transformedBoxP2x,transformedBoxP2y), (transformedBoxP3x,transformedBoxP3y), (0,0,255), 5)
#        cv2.line(boundingBox, (transformedBoxP3x,transformedBoxP3y), (transformedBoxP4x,transformedBoxP4y), (0,0,255), 5)
#        cv2.line(boundingBox, (transformedBoxP4x,transformedBoxP4y), (transformedBoxP1x,transformedBoxP1y), (0,0,255), 5)
#
#        cv2.warpPerspective(boundingBox, H, (360,640), boundingBox)
#        print(boundingBox.shape[:2])
#        print(frame2.shape[:2])
#        global frame2
#        frame2 = cv2.add(frame2, boundingBox)

#        cv2.line(frame2, (transformedBoxP1x,transformedBoxP1y), (transformedBoxP2x,transformedBoxP2y), (0,0,255), 5)
#        cv2.line(frame2, (transformedBoxP2x,transformedBoxP2y), (transformedBoxP3x,transformedBoxP3y), (0,0,255), 5)
#        cv2.line(frame2, (transformedBoxP3x,transformedBoxP3y), (transformedBoxP4x,transformedBoxP4y), (0,0,255), 5)
#        cv2.line(frame2, (transformedBoxP4x,transformedBoxP4y), (transformedBoxP1x,transformedBoxP1y), (0,0,255), 5)
        #print(retval)
        
        boundingBox = np.array([[[transformedBoxP1x,transformedBoxP1y], [transformedBoxP2x,transformedBoxP2y], [transformedBoxP3x,transformedBoxP3y], [transformedBoxP4x,transformedBoxP4y]]], np.float32)
        print("bounding",boundingBox)
        print(H)
        cv2.perspectiveTransform(boundingBox, H, boundingBox)
        print("bounding",boundingBox)
        
        cv2.line(frame2, (boundingBox[0][0][0],boundingBox[0][0][1]), (boundingBox[0][1][0],boundingBox[0][1][1]), (0,0,255), 5)
        cv2.line(frame2, (boundingBox[0][1][0],boundingBox[0][1][1]), (boundingBox[0][2][0],boundingBox[0][2][1]), (0,0,255), 5)
        cv2.line(frame2, (boundingBox[0][2][0],boundingBox[0][2][1]), (boundingBox[0][3][0],boundingBox[0][3][1]), (0,0,255), 5)
        cv2.line(frame2, (boundingBox[0][3][0],boundingBox[0][3][1]), (boundingBox[0][0][0],boundingBox[0][0][1]), (0,0,255), 5)
        
        rvec[0] = math.degrees(rvec[0])
        rvec[1] = math.degrees(rvec[1])
        rvec[2] = math.degrees(rvec[2])
        #print("rvec: ", rvec)
        #print("tvec: ", tvec)

        #cv2.putText(image, "rvec x: " + str(int(rvec[0][0])), (40, 70), cv2.FONT_HERSHEY_COMPLEX, 2,
                    #(0, 255, 0))
        #cv2.putText(image, "rvec y: " + str(int(rvec[1][0])), (40, 90), cv2.FONT_HERSHEY_COMPLEX, 2,
                    #(0, 255, 0))
        #cv2.putText(image, "rvec z: " + str(int(rvec[2][0])), (40, 110), cv2.FONT_HERSHEY_COMPLEX, 2,
                    #(0, 255, 0))

        #cv2.putText(image, "tvec x: " + str(int(tvec[0][0])), (40, 140), cv2.FONT_HERSHEY_COMPLEX, 2,
                    #(0, 255, 0))
        #cv2.putText(image, "tvec y: " + str(int(tvec[1][0])), (40, 160), cv2.FONT_HERSHEY_COMPLEX, 2,
                    #(0, 255, 0))
        #cv2.putText(image, "tvec z: " + str(int(tvec[2][0])), (40, 180), cv2.FONT_HERSHEY_COMPLEX, 2,
                    #(0, 255, 0))

        cv2.putText(image, "tvec x: " + str(int(tvec[0][0])), (40, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 255, 255))
        cv2.putText(image, "tvec y: " + str(int(tvec[1][0])), (40, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 255, 255))
        cv2.putText(image, "tvec z: " + str(int(tvec[2][0])), (40, 210), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 255, 255))


#image = cv2.resize(image, (640,360))
        outputStream.putFrame(image)
        sd.putNumber("tvec x", int(tvec[0][0]))
        sd.putNumber("tvec y", int(tvec[1][0]))
        sd.putNumber("tvec z", int(tvec[2][0]))
        global tvecXSaved, tvecYSaved, tvecZSaved
        tvecXSaved = np.append(tvecXSaved, tvec[0])
        tvecYSaved = np.append(tvecYSaved, tvec[1])
        tvecZSaved = np.append(tvecZSaved, tvec[2])
        #print("x",tvecXSaved)
        #print("y",tvecYSaved)
        #print("z",tvecZSaved)
        currentTime = time.time()
        currentRioTime = sd.getNumber("RioRuntime", "0")
        #print(currentRioTime)
        
        delay = (currentTime - frameAquiredTime)
        rioDelay = (currentRioTime - delay)
        #print(rioDelay)
        #sd.putNumber("Jetson/Pi Runtime", currentTime-startTime)
        #sd.putNumber("Frame Processing Delay", delay)
        sd.putNumber("Rio Frame Timestamp", rioDelay)
        sd.putNumber("Delay", delay)
        sd.putBoolean("Target Status", True)
	
    else:
        # pushes that it deosn't see vision target to network tables
        print("tapeDetected: ", False)
        outputStream.putFrame(image)
        sd.putNumber("tvec x", 0)
        sd.putNumber("tvec y", 0)
        sd.putNumber("tvec z", 0)
        sd.putBoolean("Target Status", False)

    cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)

    return image

cap = cv2.VideoCapture("/dev/video0")
#cap.set(3,1920)
#cap.set(4,1080)
#cap.set(3,1280)
#cap.set(4,720)
cap.set(3,640)
cap.set(4,360)
#cap.set(3,480)
#cap.set(4,270)
Low_Exposure = True
if(Low_Exposure):
	cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
	cap.set(cv2.CAP_PROP_EXPOSURE, 0.0003)
else:
	cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
	cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)

#cap2 = cv2.VideoCapture("/dev/video1")
cap2 = CameraVideoStream("/dev/video1").start()
#cap.set(3,1920)
#cap.set(4,1080)
#cap.set(3,1280)
#cap.set(4,720)
#cap2.set(3,640)
#cap2.set(4,360)
#cap.set(3,480)
#cap.set(4,270)

fps = FPS().start()

while(True):
	# Capture frame-by-frame
    #print("test2")
    ret, frame = cap.read()
    if(cap2.isOpened()):
        frame2, _ = cap2.read()
    else:
        ret, frame2 = cap.read()
    #print("test")
    frameAquiredTime = time.time()
	#rio_frameAquiredTime = sd.getNumber("RioRuntime", "0")
	#print(rio_frameAquiredTime)
    blur = cv2.blur(frame,(4,4), -1)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([50,50,40]), np.array([93,255,255]))
	#_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blur, contours, -1, (0,255,0), 1)
	#for cnt in contours:
		#approx = cv2.approxPolyDP(cnt,0.052*cv2.arcLength(cnt,True),True)
		#print len(approx)
		#if len(approx)==4:
			#print ("target detected")
			#print(approx)
			#p1 = approx[0][0]
			#p2 = approx[1][0]
			#p3 = approx[2][0]
			#p4 = approx[3][0]
			#cv2.line(blur, (p1[0],p1[1]), (p2[0],p2[1]), (255,255,255), 2)
			#cv2.line(blur, (p2[0],p2[1]), (p3[0],p3[1]), (255,0,0), 2)
			#cv2.line(blur, (p3[0],p3[1]), (p4[0],p4[1]), (255,0,0), 2)
			#cv2.line(blur, (p4[0],p4[1]), (p1[0],p1[1]), (255,0,0), 2)
			#cv2.polylines(blur, approx, True, (255,0,0), 3)
	#dst = cv2.cornerHarris(mask, 2, 3, 0.04)
	#dst = cv2.dilate(dst, None)
	#blur[dst>0.4*dst.max()]=[0,0,255]

	# Our operations on the frame come here
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Display the resulting frame

	#Thread(target=findTape(contours, blur, (image_width/2)-0.5, (image_height/2)-0.5))
    Thread(target=findTape(contours, blur, (image_width/2)-0.5, (image_height/2)-0.5, frame2))
#cv2.warpPerspective(frame2, H, (640,360), frame2)

    if(cap2.isOpened()):
        outputStream2.putFrame(frame2)

	#mask = cv2.resize(mask, (1280,720))
	#blur = cv2.resize(blur, (1280,720))
	
    #for jetson
    #Thread(target=cv2.imshow('blur',blur))
    #Thread(target=cv2.imshow('frame',mask))
    #fps.update()
    #cv2.waitKey(0)
	#cv2.imshow('harris',dst)
    print("******************")
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

# When everything done, release the capture
cap.release()
#fps.stop()
#print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#cv2.destroyAllWindows()
