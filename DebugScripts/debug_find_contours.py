import cv2 as cv
import numpy as np
import os

import params
from src.camera_controller import IDSCameraController

FRAME_WIDTH = 960    #width of captured video frames [px]
FRAME_HEIGHT = 720   #height of captured video frames [px]
WARPED_FRAME_SIDE = 900   #side length for warped frame [px]

GREEN_MIN = params.green_min
GREEN_MAX = params.green_max

RED_MIN = params.red_min
RED_MAX = params.red_max

AREA_MIN = params.area_min
AREA_MAX = params.area_max

#------------------initialize video stream-------------------
capture = IDSCameraController()
#-----------------------------------------------------------

if os.path.exists("../CalibrationData/WarpMatrix.npy"):
    M = np.load('../CalibrationData/WarpMatrix.npy')
else:
    raise RuntimeError("No 'WarpMatrix.npy'-file found. Use 'get_warp_matrix.py' first.")
while True:
    frame = capture.capture_image()
    frame_warped = cv.warpPerspective(frame, M,(WARPED_FRAME_SIDE,WARPED_FRAME_SIDE))
    frame_hsv = cv.cvtColor(frame_warped, cv.COLOR_BGR2HSV)
    mask_green = cv.inRange(frame_hsv, GREEN_MIN, GREEN_MAX)
    mask_red = cv.inRange(frame_hsv, RED_MIN, RED_MAX)
    contours_green, _ = cv.findContours(mask_green, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contours_red, _ = cv.findContours(mask_red, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    count = 0
    for i, c in enumerate(contours_green):
        area = cv.contourArea(c)
        if area < AREA_MIN or AREA_MAX < area:
          continue
        cv.drawContours(frame_warped, contours_green, i, (0, 0, 255), 2)

    for i, c in enumerate(contours_red):
        area = cv.contourArea(c)
        if area < AREA_MIN or AREA_MAX < area:
            continue
        cv.drawContours(frame_warped, contours_red, i, (0, 0, 255), 2)

    cv.imshow('Detected Contours', frame_warped)
    cv.imshow('Mask green', mask_green)
    cv.imshow('Mask red', mask_red)
    #quit while loop with key 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
             break
     
#release objects
capture.close_camera_connection()
#destroy windows
cv.destroyAllWindows()