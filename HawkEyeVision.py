import numpy as np
import cv2
import glob
import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
from pathlib import Path

def Calibrate():

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    distortion_correction_file = Path("./distortion_correction_pickle.p")
    if distortion_correction_file.is_file():
        print('Distortion correction file already created')
    else:
        # Make a list of calibration images
        images = glob.glob('./camera_cal/calibration*.jpg')
        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

import pickle
#%matplotlib inline

def Undistort():


    distortion_correction_file = Path("./distortion_correction_pickle.p")
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of test images set 1
    test_images_1 = glob.glob('./test_images/test*.jpg')
    # Make a list of test images set 2
    test_images_2 = glob.glob('./test_images/straight_lines*.jpg')
    # Combine the test images set 1 and 2
    test_images = test_images_1 + test_images_2

    # Test undistortion on an image
    test_img_path = './test_images/straight_lines1.jpg'
    img = cv2.imread(test_img_path)
    #img_size = (img.shape[1], img.shape[0])
    #print('image size is {}'.format(img_size))

    # check if we already created the calibration file with coefficients
    if distortion_correction_file.is_file():
        # load the coefficients to undistort the camera image
        with open('./distortion_correction_pickle.p', mode='rb') as f:
            calibration_file = pickle.load(f)
            mtx, dist = calibration_file['mtx'], calibration_file['dist']
    else:
        print('Calibration does not exist. Please run the cell above to create it first.')
        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open( './distortion_correction_pickle.p', 'wb' ) )

    # apply distortion correction to the camera calibration image
    filename = './camera_cal/calibration1.jpg'
    img = cv2.imread(filename)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Process the file name for saving to a different directory
    filename = filename.replace('./camera_cal/', '')
    undistorted_filename = './output_images/' + 'undist_' + filename
    cv2.imwrite(undistorted_filename, dst)
    # Convert to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=25)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=25)    

    # apply distortion correction to the test images
    for filename in test_images:
        img = cv2.imread(filename)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        # Process the file name for saving to a different directory
        filename = filename.replace('./test_images/', '')
        undistorted_filename = './output_images/' + 'undist_' + filename
        cv2.imwrite(undistorted_filename, dst)
        # Convert to RGB color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=25)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=25)

    return mtx, dist

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_uint
from skimage import io

def Warp(frame):

    # Read in a test image
    #test_img = './test_images/test5.jpg'
    # test_img = './test_images/straight_lines1.jpg'
    # test_img = './test_images/challenge.jpg'
    # test_img = './test_images/cement2.jpg'
    # test_img = './test_images/FirstFrame.jpg'

    #image = mpimg.imread(frame) # use the test image from previous cell
    #image = mpimg.imread(test_img_path)
    undistort_img = cv2.undistort(frame, mtx, dist, None, mtx)
    # global parameters
    #corners = [(576,460), (705,460), (1102, 705),(180, 705)]
    corners = [(619,392), (617,427), (663, 427),(660, 392)]
    src = np.float32([corners[0], corners[1], corners[2], corners[3]])
    #dst = np.float32([[320,0],[960, 0],[960, 720],[320, 720]])
    #dst = np.float32([[576,324],[576, 396],[704, 396],[704, 324]]) # 1st version
    #dst = np.float32([[576,324],[571, 466],[709, 466],[704, 324]])
    dst = np.float32([[624,378],[622.75, 413.5],[657.25, 413.5],[656, 378]])
    # result, result_color = pipeline(undistort_img)

    def perspective_warp(img):
        # Grab the image shape
        img_size = (img.shape[1], img.shape[0])

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, img_size)

        # Return the resulting image and matrix
        return warped

    # f, (ax3, ax4) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax3.imshow(undistort_img)
    # ax3.set_title('Original test Image', fontsize=40)

    warped_image = perspective_warp(undistort_img)
    warped_cv2_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('./warped_image.jpg', warped_cv2_image)
    # ax4.imshow(warped_image)
    # ax4.set_title('Birds Eye View', fontsize=40)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    return warped_cv2_image

import sys
import argparse
import cv2
import numpy as np
import threading
def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device", dest="video_device",
                        help="Video device # of USB webcam (/dev/video?) [0]",
                        default=0, type=int)
    arguments = parser.parse_args()
    return arguments

# On versions of L4T previous to L4T 28.1, flip-method=2
# Use the Jetson onboard camera
def open_onboard_camera():
    return cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# Open an external usb camera /dev/videoX
def open_camera_device(device_number):
    return cv2.VideoCapture(device_number)

class frameReader (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self, video_capture):
        print("test")
        ret_val, latest_frame = video_capture.read()
        print(latest_frame)
        return latest_frame

def read_cam(video_capture):
    if video_capture.isOpened():
        video_capture.set(3,1280);
        video_capture.set(4,720);
        windowName = "CannyDemo"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,1280,720)
        #cv2.resizeWindow(windowName,1920,1080)
        cv2.moveWindow(windowName,0,0)
        cv2.setWindowTitle(windowName,"Hawk Eye Vision")
        showWindow=3  # Show all stages
        showHelp = True
        font = cv2.FONT_HERSHEY_PLAIN
        helpText="'Esc' to Quit, '1' for Camera Feed, '2' for Canny Detection, '3' for All Stages. '4' to hide help"
        edgeThreshold=40
        showFullScreen = False
        frameCount = 0
        ret_val, latest_frame = video_capture.read();
        feedThread = frameReader()
        while True:
            if cv2.getWindowProperty(windowName, 0) < 0: # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break;
            # ret_val, frame = video_capture.read();
            # hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # blur=cv2.GaussianBlur(hsv,(7,7),1.5)
            # edges=cv2.Canny(blur,0,edgeThreshold)
          
            
            #if frameCount%1 == 0:
            #ret_val, latest_frame = video_capture.read();
            
            #undistort = cv2.undistort(frame, mtx, dist, None, mtx)
            latest_frame = feedThread.run(video_capture)
            
            print("test2")
            warped = Warp(latest_frame)
            print("test3")
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

            #frameCount+=1

            if showWindow == 3:  # Need to show the 4 stages
                # Composite the 2x2 window
                # Feed from the camera is RGB, the others gray
                # To composite, convert gray images to color. 
                # All images must be of the same type to display in a window
                # frameRs=cv2.resize(frame, (640,360))
                # undistortRs=cv2.resize(undistort, (640,360))
                # warpedRs=cv2.resize(warped,(1280,720))
                frameRs=cv2.resize(latest_frame, (1920,1080))
                #undistortRs=cv2.resize(undistort, (960,540))
                warpedRs=cv2.resize(warped,(1920,1080))
                #warpedRs = cv2.cvtColor(warpedRs, cv2.COLOR_RGB2BGR)
                vidBuf = np.concatenate((frameRs, warpedRs), axis=1)
                #vidBuf = np.concatenate((vidBuf, warpedRs), axis=0)
                # vidBuf = np.concatenate((frameRs, cv2.cvtColor(hsvRs,cv2.COLOR_GRAY2BGR)), axis=1)
                # blurRs=cv2.resize(blur,(640,360))
                # edgesRs=cv2.resize(edges,(640,360))
                # vidBuf1 = np.concatenate( (cv2.cvtColor(blurRs,cv2.COLOR_GRAY2BGR),cv2.cvtColor(edgesRs,cv2.COLOR_GRAY2BGR)), axis=1)
                # vidBuf = np.concatenate( (vidBuf, vidBuf1), axis=0)

            if showWindow==1: # Show Camera Frame
                displayBuf = latest_frame 
            elif showWindow == 2: # Show Canny Edge Detection
                displayBuf = warped
            elif showWindow == 3: # Show All Stages
                displayBuf = vidBuf #vidBuf

            if showHelp == True:
                cv2.putText(displayBuf, helpText, (11,20), font, 1.0, (32,32,32), 4, cv2.LINE_AA)
                cv2.putText(displayBuf, helpText, (10,20), font, 1.0, (240,240,240), 1, cv2.LINE_AA)
            cv2.imshow(windowName,displayBuf)
            print("test4")
            key=cv2.waitKey(1)
            if key == 27: # Check for ESC key
                cv2.destroyAllWindows()
                break ;
            elif key==49: # 1 key, show frame
                cv2.setWindowTitle(windowName,"Camera Feed")
                showWindow=1
            elif key==50: # 2 key, show Canny
                cv2.setWindowTitle(windowName,"Hawk Eye")
                showWindow=2
            elif key==51: # 3 key, show Stages
                cv2.setWindowTitle(windowName,"Camera, Undistorted, Hawk Eye")
                showWindow=3
            elif key==52: # 4 key, toggle help
                showHelp = not showHelp
            elif key==44: # , lower canny edge threshold
                edgeThreshold=max(0,edgeThreshold-1)
                print ('Canny Edge Threshold Maximum: ',edgeThreshold)
            elif key==46: # , raise canny edge threshold
                edgeThreshold=edgeThreshold+1
                print ('Canny Edge Threshold Maximum: ', edgeThreshold)
            elif key==74: # Toggle fullscreen; This is the F3 key on this particular keyboard
                # Toggle full screen mode
                if showFullScreen == False : 
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL) 
                showFullScreen = not showFullScreen
              
    else:
     print ("camera open failed")

if __name__ == '__main__':
    arguments = parse_cli_args()
    print("Called with args:")
    print(arguments)
    print("OpenCV version: {}".format(cv2.__version__))
    print("Device Number:",arguments.video_device)
    if arguments.video_device==0:
      video_capture=open_onboard_camera()
    else:
      video_capture=open_camera_device(arguments.video_device)
      Calibrate()
      mtx, dist = Undistort()
    read_cam(video_capture)
    video_capture.release()
    cv2.destroyAllWindows()
