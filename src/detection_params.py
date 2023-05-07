import os
import numpy as np
import src.calibration_params as calibration_params

warped_frame_side = calibration_params.warped_frame_side

if os.path.exists("../CalibrationData/ReferenceAxis.npy"):
    reference_axis = np.load('../CalibrationData/ReferenceAxis.npy')
else:
    reference_axis = np.array(([int(warped_frame_side / 2), int(warped_frame_side / 2)], [0, warped_frame_side / 2]))
    print("WARNING: No 'ReferenceAxis.npy'-file found. Use 'get_reference_axis.py'-script for calibration. "
          "Continue with default values.")

pos_A = reference_axis[0]
vec_ref_1 = reference_axis[1]

if os.path.exists("../CalibrationData/WarpMatrix.npy"):
    warp_matrix = np.load('../CalibrationData/WarpMatrix.npy')
else:
    warp_matrix = np.array([[8.58769289e-01, -1.08283228e-02, -3.82004069e+02],
                            [2.48709061e-03, 8.58046261e-01, -8.09075447e+01],
                            [-7.62754274e-06, -1.20314697e-05, 1.00000000e+00]])
    print("WARNING: No 'WarpMatrix.npy'-file found. Use 'get_warp_matrix.py'-script to compute warp matrix. "
          "Continue with default matrix.")

# Threshold of searched colour red in HSV space
if os.path.exists("../CalibrationData/HsvMinMaxRed.npy"):
    red_min_max = np.load('../CalibrationData/HsvMinMaxRed.npy')
    red_min = red_min_max[0, :]
    red_max = red_min_max[1, :]
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
else:
    green_min = np.array([55, 105, 25])
    green_max = np.array([65, 125, 45])
    print("WARNING: No 'HsvMinMaxGreen.npy'-file found. Use 'get_hsv_masks.py'-script to collect colour values. "
          "Continue with default values.")

# Filter size of contours
area_min = 0.00145 * pow(warped_frame_side, 2)
area_max = 0.00185 * pow(warped_frame_side, 2)


visu_axis_length = 100
