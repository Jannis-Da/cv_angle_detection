from camera_controller import IDSCameraController
import os.path
import numpy as np
from angle_detection import AngleDetector


def main():
    camera = IDSCameraController(param_file=r"../CameraParameters/cp_AngleDetection.ini")
    detection = AngleDetector(camera, definition=0)
    frame_count = 0
    print("Checking for references. Do not move the pendulum.")
    while np.isnan(detection.pos_B).all() or np.isnan(detection.pos_C).all():
        frame_count += 1
        if frame_count >= 1000:
            camera.close_camera_connection()
            raise RuntimeError(f"Reference points not found after {frame_count} frames. Check conditions.")
        detection.get_angle()
        vec_bc = detection.pos_C - detection.pos_B
        pos_a = detection.pos_B - vec_bc
        ref_vec = detection.pos_B - pos_a
        reference_values = np.array((pos_a, ref_vec))

    if not os.path.exists(r"../CalibrationData"):
        os.makedirs(r"../CalibrationData")

    np.save('../CalibrationData/ReferenceAxis.npy', reference_values)
    camera.close_camera_connection()

    print("Reference calibration finished successfully.")


if __name__ == '__main__':
    main()