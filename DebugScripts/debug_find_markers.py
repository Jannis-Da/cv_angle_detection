import cv2 as cv
import src.detection_params
import src.calibration_params
from src.camera_controller import IDSCameraController

camera = IDSCameraController(param_file=r"../CameraParameters/cp_DetectMarker.ini")

while True:
    
    frame = camera.capture_image()
    corners, ids, _ = cv.aruco.detectMarkers(frame, src.calibration_params.aruco_dict,
                                             parameters=src.calibration_params.aruco_params)
    cv.aruco.drawDetectedMarkers(frame, corners)
    cv.namedWindow('Show Marker', cv.WINDOW_NORMAL)
    cv.imshow('Show Marker', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
         
# After the loop release the cap object
camera.close_camera_connection()
# Destroy all the windows
cv.destroyAllWindows()
