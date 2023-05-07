import get_warp_matrix
import get_hsv_masks
import get_reference_axis
import calibration_params as cal_params
from camera_controller import IDSCameraController
from angle_detection import AngleDetector
import cv2 as cv
import numpy as np


def main():
    print("Start system calibration.")
    get_warp_matrix.main()
    get_hsv_masks.main()
    blank = np.zeros((50, 550, 3), np.uint8)
    label = "HSV mask calibration finished. Stop pendulum and press 'q' to continue calibration."
    cv.putText(blank, label, (2, 25), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

    while True:
        cv.imshow("User instruction", blank)
        if cv.waitKey(1) & 0xff == ord('q'):
            break

    cv.destroyAllWindows()

    get_reference_axis.main()
    print("System calibration finished successfully.")

    # Initialize camera, param_file = set camera parameters with .ini file
    camera = IDSCameraController(param_file=r"../CameraParameters/cp_AngleDetection.ini")

    # Initialize AngleDetector object,
    # definition = set definition of angle (0 = second angle relative to the first, 1=second angle absolute)
    measurement = AngleDetector(camera, definition=0)

    while True:
        measurement.get_angle()
        measurement.get_angular_vel()

        cv.rectangle(measurement.visu, (0, 0), (cal_params.warped_frame_side, 80), (255, 255, 255), -1)
        label = "Check calibration result. Use the debug scripts to track possible deviations." \
                " Press 'q' to close window."
        cv.putText(measurement.visu, label, (cal_params.warped_frame_side-670, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)
        measurement.visualize()

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all resources
    # Close visualization
    cv.destroyAllWindows()

    # End camera connection
    camera.close_camera_connection()

if __name__ == '__main__':
    main()