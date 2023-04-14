import os
import numpy as np
import cv2 as cv

warped_frame_side = 900
pos_A = np.array([int(warped_frame_side / 2), int(warped_frame_side / 2)])

# Threshold of searched colour red in HSV space
if os.path.exists("../CalibrationData/HsvMinMaxRed.npy"):
    red_min_max = np.load('../CalibrationData/HsvMinMaxRed.npy')
    red_min = red_min_max[0, :]
    red_max = red_min_max[1, :]
elif not os.path.exists("../CalibrationData/WarpMatrix.npy"):
    red_min = np.array([0, 180, 55])
    red_max = np.array([15, 210, 75])
else:
    red_min = np.array([0, 180, 55])
    red_max = np.array([15, 210, 75])
    print("WARNING: No 'HsvMinMaxRed.npy'-file found. Use 'get_hsv_masks.py'-script to collect colour values. "
          "Continue with default values.")

# Threshold of searched colour green in HSV space
if os.path.exists("../CalibrationData/HsvMinMaxGreen.npy"):
    green_min_max = np.load('../CalibrationData/HsvMinMaxGreen.npy')
    green_min = green_min_max[0, :]
    green_max = green_min_max[1, :]
elif not os.path.exists("../CalibrationData/WarpMatrix.npy"):
    green_min = np.array([55, 105, 25])
    green_max = np.array([65, 125, 45])
else:
    green_min = np.array([55, 105, 25])
    green_max = np.array([65, 125, 45])
    print("WARNING: No 'HsvMinMaxGreen.npy'-file found. Use 'get_hsv_masks.py'-script to collect colour values. "
          "Continue with default values.")

# Filter size of contours
area_min = 1000
area_max = 1400


visu_axis_length = 100

# define used aruco markers
aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_250)
aruco_params = cv.aruco.DetectorParameters_create()
