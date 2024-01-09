from utils import *
from pathlib import Path
import os
import cv2
import numpy as np
import mediapipe as mp
import glob
import pyrealsense2 as rs
from tqdm import tqdm
class Action:
    def __init__(self, opt_path):
        self.opt = read_yaml(opt_path)
        self.mp_pose = mp.solutions.pose
        self.posedetector = self.mp_pose.Pose()
        self.count = 0
        self.is_action = None
        self.num_num_sub_video = 0
        self.w,self.h = self.opt['image'].values()

    def new_start(self, file_path):
        self.pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, file_path, repeat_playback=False)
        self.pipeline.start(config)

        self.count = 0
        self.is_action = None
        self.num_sub_video = 0
        name_action = os.path.basename(os.path.dirname(file_path))
        self.folder_output = os.path.join(self.opt['out_folder'], name_action)
        os.makedirs(self.folder_output, exist_ok=True)
        self.input_id = os.path.basename(file_path).split(".")[0]

    def get_key_points(self, frame):
        h_im, _, _ = frame.shape
        results = self.posedetector.process(frame)
        left_y, right_y, centre = None,None,None
        if results.pose_landmarks:
            left_y = results.pose_landmarks.landmark[15].y * h_im
            right_y = results.pose_landmarks.landmark[16].y * h_im
            centre = results.pose_landmarks.landmark[24].y * h_im - \
                    (results.pose_landmarks.landmark[24].y * h_im - results.pose_landmarks.landmark[12].y * h_im) * self.opt['rate_up']
        return left_y, right_y, centre, results

    def start_capture(self):
        name = f"{self.input_id}_sub_{self.num_sub_video}.mp4"
        output_file = os.path.join(self.folder_output, name)
        self.num_sub_video += 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_file, fourcc, self.opt['out_fps'], (self.w, self.h))
        if self.pbar2:
            self.pbar2.set_description(f"start caputure: {output_file}")

    def check_action(self, left_y, right_y, centre):
        return left_y < centre or right_y < centre

    def update_action(self, is_action, frame):
        if self.is_action:
                self.out.write(frame)
        if is_action:
            self.count = max(1, self.count + 1)
            if self.count >= self.opt['thresh_frame_action']:
                if self.is_action is False or self.is_action is None:
                    self.start_capture()
                self.is_action = True
        else:
            self.count = min(-1, self.count - 1)
            if self.count <= -self.opt['thresh_frame_no_action']:
                if self.is_action:
                    self.out.release()
                self.is_action = False

    def add_new_frame(self, frame):
        left_y, right_y, centre, results = self.get_key_points(frame)
        frame = cv2.resize(frame,(self.w,self.h))
        if left_y is not None:
            is_action = self.check_action(left_y, right_y, centre)
            self.update_action(is_action, frame)
        else:
            self.update_action(self.is_action, frame)
            
    def segment_video(self):
        count = 0
        while True:
            try:
                frames = self.pipeline.wait_for_frames()
                count += 1 
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
                self.add_new_frame(rgb_image)
            except Exception as e:
                break
        # self.pipeline.release()
    def segment_video_folder(self):
        count = 0
        self.pbar1 = tqdm(glob.glob(os.path.join(self.opt['input_folder'], "*")), desc='Processing Subfolders')
        for i,sub_folder in enumerate(self.pbar1):
            with open("last.txt",'w') as  f:
                f.write(str(i+1))
            count+=1
            if count<self.opt['start']:
                continue
            if count>self.opt['end']:
                break
            self.pbar2 = tqdm(glob.glob(os.path.join(sub_folder, "*")), desc=f'Processing Files in {os.path.basename(sub_folder)}', leave=False)
            for path in self.pbar2:
                self.new_start(path)
                self.segment_video()
            
                
       
            


# Assuming you have a valid opt.yaml file and rs is the RealSense module
# You should also have a loop to capture frames and call add_new_frame method
# Example:
action = Action(r"D:\thu_anh\config.yaml")
action.segment_video_folder()