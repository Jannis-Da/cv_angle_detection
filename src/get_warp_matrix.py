import cv2 as cv
import os.path
import numpy as np
import calibration_params as cal_params
from camera_controller import IDSCameraController


class WarpMatrixCalculator:
    def __init__(self):
        self.camera = IDSCameraController(param_file=r"../CameraParameters/cp_DetectMarker.ini")
        if not os.path.exists(r"../CalibrationData"):
            os.makedirs(r"../CalibrationData")

    def get_matrix(self):
        print("Calculate warp matrix...")
        frame_count = 0
        # search for aruco markers in frames until all 4 markers are found or max values for frames is reached
        while True:
            frame_count += 1
            frame = self.camera.capture_image()
            corners, ids, _ = cv.aruco.detectMarkers(frame, cal_params.aruco_dict, parameters=cal_params.aruco_params)
            if contains_zero_to_three(ids):
                break
            elif frame_count >= 1000:
                if ids is None:
                    ids_count = 0
                else:
                    ids_count = len(ids)
                self.camera.close_camera_connection()
                raise RuntimeError(f"Found only {ids_count} of 4 markers after {frame_count} frames. Check conditions.")
        warp_matrix = calc_matrix(corners, ids)
        np.save('../CalibrationData/WarpMatrix.npy', warp_matrix)
        self.camera.close_camera_connection()
        print("Calculation of warp matrix finished successfully.")


# Check if all Markers (IDs 0-3) were found
def contains_zero_to_three(ids):
    if ids is None:
        return False
    return np.isin([0, 1, 2, 3], ids).all()


# Calculation of warp Matrix warp_matrix
def calc_matrix(corners, ids):
    # Find index of specific markers in ID-List
    marker_idxs = [np.where(ids == i)[0][0] for i in range(4)]

    # Extract corner points-coordinates corresponding to the four markers with the desired IDs
    up_left = corners[marker_idxs[0]][0][2]
    up_right = corners[marker_idxs[1]][0][3]
    down_right = corners[marker_idxs[2]][0][3]
    down_left = corners[marker_idxs[3]][0][2]

    # Define source and destination points for perspective transform
    src_pts = np.float32([up_left, up_right, down_right, down_left])
    dst_pts = np.float32([[0, 0], [cal_params.warped_frame_side, 0], [cal_params.warped_frame_side, cal_params.warped_frame_side],
                          [0, cal_params.warped_frame_side]])

    # Compute perspective transform matrix
    warp_matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
    return warp_matrix


def main():
    matrix_calc = WarpMatrixCalculator()
    matrix_calc.get_matrix()


if __name__ == '__main__':
    main()
