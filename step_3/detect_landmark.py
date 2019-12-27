"""
    using dlib to detected face landmark of albedo
    need python3, so use pytorch_3DDFA environment
"""
import os
import sys
sys.path.append('../useful_code/3DDFA/')

import numpy as np
import cv2
import dlib
from utils.inference import draw_landmarks

baseFolder = '../useful_code/3DDFA/'
class detect_landmark():
    def __init__(self):
        # load dlib model for face detection and landmark  for face cropping
        dlib_landmark_model = os.path.join(baseFolder, 'models/shape_predictor_68_face_landmarks.dat')
        self.face_regressor = dlib.shape_predictor(dlib_landmark_model)
        self.face_detector = dlib.get_frontal_face_detector()

    def detect(self, img):
        '''
            detect landmarks of input img
        '''
        rects = self.face_detector(img, 1)
        if len(rects) == 0:
            return
        rect = rects[0]
        pts = self.face_regressor(img, rect).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]).T  # detected landmarks, should record this

        return pts

    def draw_landmark(self, pts, img, saveName):
        '''
            draw landmark on image and save it as saveName
        '''
        pts_res = []
        pts_res.append(pts)
        draw_landmarks(img, pts_res, wfp=saveName, show_flg=False)
        return 

    def save_landmark(self, pts, saveName):
        '''
            save pts to file
        '''
        np.savetxt(saveName, pts, fmt='%.3f')
        return
