import cv2 as cv
import src.params
from src.camera_controller import IDSCameraController

"""""
capture = cv.VideoCapture(1, cv.CAP_DSHOW)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 960)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
capture.set(cv.CAP_PROP_BRIGHTNESS, 40.0)
"""""

capture = IDSCameraController(param_file=r"../CameraParameters/cp_DetectMarker.ini")

while True:
    
    frame = capture.capture_image()

    corners, ids, _ = cv.aruco.detectMarkers(frame, src.params.aruco_dict, parameters=src.params.aruco_params)
    
    cv.aruco.drawDetectedMarkers(frame, corners)

    cv.imshow('Show Marker', frame)
    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
         
# # After the loop release the cap object
capture.close_camera_connection()
# # Destroy all the windows
cv.destroyAllWindows()