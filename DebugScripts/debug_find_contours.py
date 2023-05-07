import cv2 as cv
import src.detection_params as params
from src.camera_controller import IDSCameraController


capture = IDSCameraController()

while True:
    frame = capture.capture_image()
    frame_warped = cv.warpPerspective(frame, params.warp_matrix, (params.warped_frame_side, params.warped_frame_side))
    frame_hsv = cv.cvtColor(frame_warped, cv.COLOR_BGR2HSV)
    mask_green = cv.inRange(frame_hsv, params.green_min, params.green_max)
    mask_red = cv.inRange(frame_hsv, params.red_min, params.red_max)
    contours_green, _ = cv.findContours(mask_green, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contours_red, _ = cv.findContours(mask_red, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    count = 0
    for i, c in enumerate(contours_green):
        area = cv.contourArea(c)
        if area < params.area_min or params.area_max < area:
          continue
        cv.drawContours(frame_warped, contours_green, i, (0, 0, 255), 2)

    for i, c in enumerate(contours_red):
        area = cv.contourArea(c)
        if area < params.area_min or params.area_max < area:
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