import cv2 as cv
import numpy as np
import math
import params
import time
import os


def norm_vector(vector):
    """
    Returns the normalized version of the input vector.

    Parameters
    ----------
    vector : numpy.ndarray
        An array representing a vector.

    Returns
    -------
    numpy.ndarray
        A normalized vector with the same direction as the input vector.
    """
    return vector / np.linalg.norm(vector)


def calc_angle(v1, v2):
    """
    Calculates the angle in radians between two vectors.

    Parameters
    ----------
    v1 :  numpy.ndarray
        A numpy array representing the first vector.
    v2 :  numpy.ndarray
        A numpy array representing the second vector.

    Returns
    -------
    float
        The angle in radians between the two vectors.
    """
    # Normalize vectors
    v1_u = norm_vector(v1)
    v2_u = norm_vector(v2)

    # Dot-product of normalized vectors, limited to values between -1 and 1, calculate angle with arccos
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def check_side(point_a, point_b, point_c):
    """
    Determines on which side of the line between points A and B the point C is located.

    Parameters
    ----------
    point_a : numpy.ndarray
        Coordinates of point A in the form [x, y].
    point_b : numpy.ndarray
        Coordinates of point B in the form [x, y].
    point_c : numpy.ndarray
        Coordinates of point C in the form [x, y].

    Returns
    -------
    int
        -1 if C is on the right side of the line, +1 if C is on the left side, 0 if C is on the line.
    """
    # Calculate determinant of 2x2-matrix built with the 3 points
    position = np.sign((point_b[0] - point_a[0]) * (point_c[1] - point_a[1]) - (point_b[1] - point_a[1])
                       * (point_c[0] - point_a[0]))
    return position


def get_center(pts):
    """
    Calculates the center point of a contour.

    Parameters
    ----------
    pts : numpy.ndarray
        A list of points that make up the contour.

    Returns
    -------
    numpy.ndarray
        A numpy array representing the center point of the contour.
    """
    moments = cv.moments(pts)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    center = np.array([cx, cy])
    return center


def get_axis_visu(axis):
    """
    Computes the visual representation of an axis vector for display.

    Parameters
    ----------
    axis : numpy.ndarray
        A vector representing an axis.

    Returns
    -------
    numpy.ndarray
        A vector representing the visual representation of the input axis, with selected length and integer coordinates.
    """
    axis = norm_vector(axis) * params.visu_axis_length
    return axis.astype(int)


def get_center_position(contours):
    """
    Computes the center of the contour that is searched for.

    Parameters
    ----------
    contours : numpy.ndarray
        A list of contours.

    Returns
    -------
    position : numpy.ndarray
        The center point of the searched for contour.  If the contour was not found, the position is set to NaN.
    found : bool
        A boolean value, indicating if the contour was found.
    """
    contours_cnt = 0
    position = np.zeros(2)
    for i, c in enumerate(contours):
        # Filter contours by size of the area they are covering to make sure only coloured circles are detected
        area = cv.contourArea(c)
        if area < params.area_min or params.area_max < area:
            continue

        # safe center of contour
        position = get_center(c)
        contours_cnt += 1

    # Make sure that only one closed contour is found by checking counter
    if contours_cnt != 1:
        position = np.full(2, np.nan)
        found = False
    else:
        found = True
    return position, found


class AngleDetector:
    """
    Class for detecting the angles and angular velocities of a double pendulum system.

    Parameters
    ----------
    camera : Camera
        The camera object used for capturing the video stream.
    definition : int, optional
        An integer value that determines the reference vector for the second pendulum arm.
        0 for absolute, 1 for relative measurement.

    Attributes
    ----------
    warp_matrix : numpy.ndarray
        The warp matrix used for warping the captured frames.
    angle1 : float
        The angle of the first pendulum arm. [rad]
    angle2 : float
        The angle of the second pendulum arm. [rad]
    angular_vel1 : float
        The angular velocity of the first pendulum arm. [rad/s]
    angular_vel2 : float
        The angular velocity of the second pendulum arm. [rad/s]
    angle_buffer_1 : AngleBuffer
        The buffer used for storing the previous angle values of the first pendulum arm.
    angle_buffer_2 : AngleBuffer
        The buffer used for storing the previous angle values of the second pendulum arm.
    start_time : float
        The timestamp of when the AngleDetector object was created.
    timestamp : float
        The timestamp of the most recent angle calculation.
    definition : int
        An integer value that determines the reference vector for the second pendulum arm.
    visu : numpy.ndarray
        The most recent warped frame.
    visu_used : Bool
        A boolean indicator to show if visu function is used.
    pos_A : numpy.ndarray
        The position of the fixed pivot point of the double pendulum.
    pos_B : numpy.ndarray
        The position of the second pivot point.
    pos_C : numpy.ndarray
        The position of the end of the second pendulum arm.
    contours_red : numpy.ndarray
        The contours of the red objects in the most recent frame.
    contours_green : numpy.ndarray
        The contours of the green objects in the most recent frame.
    vec_ref_1 : numpy.ndarray
        The reference vector used for the first pendulum arm.
    vec_ref_2 : numpy.ndarray
        The reference vector used for the second pendulum arm.

    Methods
    -------
    get_contours()
        Filters the captured frame for red and green colour and extracts the contours separately.
    get_angle()
        Calculates the angles of the double pendulum using the extracted contours.
    get_angular_vel()
        Calculates the angular velocity with the values in the two angle buffers.
    visualize(vis_text=True, vis_contours=True, vis_vectors=True)
        Visualizes the live results of angle detection.
    """
    def __init__(self, camera, definition=0):
        self.camera = camera
        self.definition = definition
        if os.path.exists("../CalibrationData/WarpMatrix.npy"):
            self.warp_matrix = np.load('../CalibrationData/WarpMatrix.npy')
        else:
            raise RuntimeError("No 'WarpMatrix.npy'-file found. Use 'get_warp_matrix.py' to compute warp matrix.")
        self.contours_red = None
        self.contours_green = None
        self.pos_A = params.pos_A
        self.pos_B = np.full(2, np.nan)
        self.pos_C = np.full(2, np.nan)
        self.vec_ref_1 = np.array([0, params.warped_frame_side / 2], dtype=int)
        self.vec_ref_2 = np.zeros(2, dtype=int)
        self.angle1 = float("NaN")
        self.angle2 = float("NaN")
        self.angular_vel1 = float("NaN")
        self.angular_vel2 = float("NaN")
        self.angle_buffer_1 = AngleBuffer()
        self.angle_buffer_2 = AngleBuffer()
        self.start_time = time.time()
        self.timestamp = float("NaN")
        self.visu = None
        self.visu_used = False

    def get_contours(self):
        """
        Filters the captured frame for red and green colour and extracts the contours separately

        Returns
        -------
        contours_red : numpy.ndarray
            The red contours found in the frame.
        contours_green: numpy.ndarray
            The green contours found in the frame.
        """
        frame = self.camera.capture_image()

        # Warp frame with warp matrix to frame with defined side length
        frame_warped = cv.warpPerspective(frame, self.warp_matrix, (params.warped_frame_side, params.warped_frame_side))
        self.visu = frame_warped

        # Convert frame to hsv-colour-space for colour filtering
        frame_hsv = cv.cvtColor(frame_warped, cv.COLOR_BGR2HSV)

        # Preparing mask to overlay
        mask_red = cv.inRange(frame_hsv, params.red_min, params.red_max)
        mask_green = cv.inRange(frame_hsv, params.green_min, params.green_max)

        # Find contours for red and green shapes in frame
        contours_red, _ = cv.findContours(mask_red, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        contours_green, _ = cv.findContours(mask_green, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        return contours_red, contours_green

    def get_angle(self):
        """
        Calculates the angles of the double pendulum using the extracted contours.

        Returns
        -------
        list
            A list of two float elements representing the first and second angle respectively.
        """
        # Try to detect red and green circle in frame and calculate center
        self.contours_red, self.contours_green = self.get_contours()
        self.pos_B, found_B = get_center_position(self.contours_red)
        self.pos_C, found_C = get_center_position(self.contours_green)

        # Calculate angle of first arm
        if found_B:
            vec_ab = self.pos_B - self.pos_A
            # Check in which rotational direction angle should be measured
            if check_side(self.pos_A, self.pos_A + self.vec_ref_1, self.pos_B) == (-1 | 0):
                self.angle1 = calc_angle(self.vec_ref_1, vec_ab)
            else:
                self.angle1 = -calc_angle(self.vec_ref_1, vec_ab)
        else:
            # Set value to NaN when no matching red contour could be found
            self.angle1 = float('NaN')

        # Calculate angle of second arm
        if found_B & found_C:
            vec_bc = self.pos_C - self.pos_B
            # Check for chosen angle definition
            if self.definition == 0:
                self.vec_ref_2 = self.pos_B-self.pos_A
                # Check in which rotational direction angle should be measured
                if check_side(self.pos_B, self.pos_B + self.vec_ref_2, self.pos_C) == (-1 | 0):
                    self.angle2 = calc_angle(self.vec_ref_2, vec_bc)
                else:
                    self.angle2 = -calc_angle(self.vec_ref_2, vec_bc)
            elif self.definition == 1:
                self.vec_ref_2 = np.array([0, 150], dtype=int)
                # Check in which rotational direction angle should be measured
                if check_side(self.pos_B, self.pos_B + self.vec_ref_2, self.pos_C) == (-1 | 0):
                    self.angle2 = calc_angle(self.vec_ref_2, vec_bc)
                else:
                    self.angle2 = -calc_angle(self.vec_ref_2, vec_bc)
        else:
            # Set Value to Nan when one of the contours could not be found (calculating second angle would be
            # impossible without position of point B
            self.angle2 = float('NaN')

        # Fill angle buffer for calculation of angular velocitys
        self.timestamp = time.time()-self.start_time
        self.angle_buffer_1.shift_buffer(self.angle1, self.timestamp)
        self.angle_buffer_2.shift_buffer(self.angle2, self.timestamp)

        return [self.angle1, self.angle2]

    def get_angular_vel(self):
        """
        Calculates the angular velocity with the values in the two angle buffers.

        Returns
        -------
        list
            A list of two float elements representing the angular velocity of the
            first and second pendulum arm respectively.
        """
        if not np.all(self.angle_buffer_1.timestamps == 0):
            self.angular_vel1 = self.angle_buffer_1.calc_velocity()
            self.angular_vel2 = self.angle_buffer_2.calc_velocity()
            return [self.angular_vel1, self.angular_vel2]
        else:
            raise RuntimeError("No values for velocity calculation available. "
                               "Use 'get_angular_vel()' only in combination with 'get_angle()'-function.")

    def visualize(self, vis_text=True, vis_contours=True, vis_vectors=True):
        """
        Visualizes the live results of angle detection.

        Parameters
        ----------
        vis_text : bool
            Boolean value to decide if text should be visualized.
        vis_contours : bool
            Boolean value to decide if contours should be visualized.
        vis_vectors : bool
            Boolean value to decide if vectors should be visualized.
        """
        if self.visu is not None:
            self.visu_used = True
            if vis_contours:
                for i, c in enumerate(self.contours_red):
                    area = cv.contourArea(c)
                    if area < params.area_min or params.area_max < area:
                        continue
                    cv.drawContours(self.visu, self.contours_red, i, (0, 0, 255), 2)
                for i, c in enumerate(self.contours_green):
                    area = cv.contourArea(c)
                    if area < params.area_min or params.area_max < area:
                        continue
                    cv.drawContours(self.visu, self.contours_green, i, (0, 0, 255), 2)

            if not math.isnan(self.angle1):
                label11 = " angle1  = " + str(int(np.rad2deg(self.angle1))) + " deg"
                if not math.isnan(self.angular_vel1):
                    label12 = " velocity1 = " + str(int(np.rad2deg(self.angular_vel1))) + " deg/s"
                else:
                    label12 = " velocity1 = NaN"
                if vis_vectors:
                    # print vector between joint A and joint B
                    cv.line(self.visu, self.pos_A, self.pos_B, (255, 0, 0), thickness=1)
                    # print angle visu at joint A
                    cv.line(self.visu, self.pos_A, self.pos_A + get_axis_visu(self.vec_ref_1), (0, 255, 0),
                            lineType=cv.LINE_8, thickness=1)  # reference axis
                    cv.ellipse(self.visu, self.pos_A, (45, 45), 0, 90, 90 - np.rad2deg(self.angle1),
                               (0, 255, 0))  # angle ellipse
            else:
                label11 = " angle1 = NaN"
                label12 = " velocity1 = NaN"
            if not math.isnan(self.angle2):
                label21 = " angle2 = " + str(int(np.rad2deg(self.angle2))) + " deg"
                if not math.isnan(self.angular_vel2):
                    label22 = " velocity2 = " + str(int(np.rad2deg(self.angular_vel2))) + " deg/s"
                else:
                    label22 = " velocity2 = NaN"
                if vis_vectors:
                    # print  vectors between joint B and tip
                    cv.line(self.visu, self.pos_B, self.pos_C, (255, 0, 0), thickness=1)
                    # print angle visu at joint B
                    if self.definition == 0:  # differ between absolute and relative angles
                        cv.line(self.visu, self.pos_B, self.pos_B + get_axis_visu(self.vec_ref_2), (0, 255, 0),
                                lineType=cv.LINE_8, thickness=1)  # reference axis
                        cv.ellipse(self.visu, self.pos_B, (45, 45), 0, 90 - np.rad2deg(self.angle1),
                                   90 - np.rad2deg(self.angle1) - np.rad2deg(self.angle2), (0, 255, 0))  # angle ellipse
                    elif self.definition == 1:
                        cv.line(self.visu, self.pos_B, self.pos_B + get_axis_visu(self.vec_ref_2),
                                (0, 255, 0), lineType=cv.LINE_8,
                                thickness=1)  # reference axis
                        cv.ellipse(self.visu, self.pos_B, (45, 45), 0, 90, 90 - np.rad2deg(self.angle2),
                                   (0, 255, 0))  # angle ellipse
            else:
                label21 = " angle2 = NaN"
                label22 = " velocity2 = NaN"

            if vis_text:
                cv.rectangle(self.visu, (0, 0), (230, 80), (255, 255, 255), -1)
                cv.putText(self.visu, label11, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(self.visu, label12, (0, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(self.visu, label21, (0, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(self.visu, label22, (0, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

            if vis_vectors:
                cv.circle(self.visu, self.pos_A, 2, (255, 0, 0), thickness=3)
                if not math.isnan(self.pos_B[0]):
                    cv.circle(self.visu, self.pos_B, 2, (255, 0, 0), thickness=3)
                if not math.isnan(self.pos_C[0]):
                    cv.circle(self.visu, self.pos_C, 2, (255, 0, 0), thickness=3)

            # show frame in pop-up window
            cv.namedWindow('Angle Detection', cv.WINDOW_AUTOSIZE)
            cv.imshow('Angle Detection', self.visu)
        else:
            raise RuntimeError("Nothing to visualize. "
                               "Use 'visualize()'-function only in combination with 'get_angle()'-function.")


class AngleBuffer:
    def __init__(self):
        self.timestamps = np.zeros(2)
        self.angles = np.zeros(2)

    def shift_buffer(self, angle, timestamp):
        """
        Shifts the angle buffer by replacing the oldest angle and timestamp with the newest ones.

        Parameters
        ----------
        angle : float
            The newest angle value to add to the buffer.
        timestamp : float
            The corresponding timestamp of the newest angle value.
        """
        self.timestamps[0] = self.timestamps[1]
        self.angles[0] = self.angles[1]
        self.timestamps[1] = timestamp
        self.angles[1] = angle

    def calc_velocity(self):
        """
        Calculates the angular velocity based on the difference between two angle values
        and corresponding timestamps in the angle buffer.

        Returns
        -------
        float
            The calculated velocity as a float value in radians per seconds.
        """
        return (self.angles[1] - self.angles[0])/(self.timestamps[1] - self.timestamps[0])
