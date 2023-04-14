import os.path
import datetime
import params
import cv2 as cv
import pandas as pd


class VisuRecorder:
    def __init__(self, rec_filename="rec"):

        if not os.path.exists("../VisuRecords"):
            os.makedirs("../VisuRecords")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.recorder = cv.VideoWriter(f"../VisuRecords/{timestamp}_{rec_filename}.avi",
                                       cv.VideoWriter_fourcc(*'MJPG'), 30,
                                       (params.warped_frame_side, params.warped_frame_side))

    def record_visu(self, angle_detector):
        if angle_detector.visu is not None and angle_detector.visu_used is True:
            self.recorder.write(angle_detector.visu)
        elif angle_detector.visu_used is False:
            raise RuntimeError("No visualization found to be recorded. "
                               "Use 'record_visu()'-function only in combination with 'visualize()-function.")
        else:
            raise RuntimeError("No visualization found to be recorded. "
                               "Use 'record_visu()'-function only in combination with 'get_angle()'-function")

    def stop_recording_visu(self):
        self.recorder.release()


class DataRecorder:
    def __init__(self, log_filename="log"):
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if not os.path.exists("../DataRecords"):
            os.makedirs("../DataRecords")

        self.filename = log_filename
        self.df = pd.DataFrame({"Time": [], "Angle1": [], "Angle2": [], "AngularVel1": [], "AngularVel2": []})

    def write_datarow(self, angle_detector):
        if angle_detector.visu is not None:
            new_row = pd.Series(
                {"Time": angle_detector.timestamp, "Angle1": angle_detector.angle1, "Angle2": angle_detector.angle2,
                 "AngularVel1": angle_detector.angular_vel1, "AngularVel2": angle_detector.angular_vel2})
            self.df = pd.concat([self.df, new_row.to_frame().T], ignore_index=True)
        else:
            raise RuntimeError("No data found to be saved. "
                               "Use 'write_datarow()'-function only in combination with 'get_angle()'-function")

    def save_pickle(self):
        if len(self.df.index) != 0:
            self.df.to_pickle(f"../DataRecords/{self.timestamp}_{self.filename}.pkl")
        else:
            print("WARNING: No values found to save to .pkl-file. Use 'write_datarow'-function to collect data.")

    def save_csv(self):
        if len(self.df.Time.value_counts()) > 0:
            self.df.to_csv(f"../DataRecords/{self.timestamp}_{self.filename}.csv", index=False, decimal=',')
        else:
            print("WARNING: No values found to save to .csv-file. Use 'write_datarow'-function to collect data.")

class FrameExtractor:
    def __init__(self, frame_filename='frame', folder="folder", rate=10, count=10):
        self.filename = frame_filename
        self.folder = folder
        self.rate = rate
        self.frames = count
        self.rate_count = 0
        self.frame_count = 1

        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%warp_matrix-%S")

        if not os.path.exists(f"../ExtractedFrames/{self.timestamp}_{self.folder}"):
            os.makedirs(f"../ExtractedFrames/{self.timestamp}_{self.folder}")

    def extract_frames(self, angle_detector):
        if angle_detector.visu is not None:
            self.rate_count += 1
            if (self.rate_count >= self.rate) & (self.frame_count <= self.frames):
                cv.imwrite(f"../ExtractedFrames/{self.timestamp}_{self.folder}/{self.filename}_{self.frame_count}.jpg", angle_detector.visu)
                self.frame_count += 1
                self.rate_count = 0
        elif angle_detector.visu_used is False:
            raise RuntimeError("No visualization found to extract frames from. "
                               "Use 'extract_frames()'-function only in combination with 'visualize()-function.")
        else:
            raise RuntimeError("No visualization found to extract frames from. "
                               "Use 'extract_frames()'-function only in combination with 'get_angle()'-function")
