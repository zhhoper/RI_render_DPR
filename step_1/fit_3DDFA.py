#!/usr/bin/env python3
# coding: utf-8

#__author__ = 'cleardusk'
__author__ = 'Hao'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

adapated from 3DDFA of Xiangyu Zhu

"""
import os
import sys
sys.path.append('../useful_code/3DDFA/')

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn

STD_SIZE = 120

baseFolder = '../useful_code/3DDFA/'
class fit_3DDFA():
    def __init__(self, mode='gpu'):
        # load model
        checkpoint_fp = os.path.join(baseFolder, 'models/phase1_wpdc_vdc_v2.pth.tar')
        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        self.model = getattr(mobilenet_v1, 'mobilenet_1')(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

        model_dict = self.model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        self.model.load_state_dict(model_dict)
        self.mode = mode
        if self.mode == 'gpu':
            cudnn.benchmark = True
            self.model = self.model.cuda()
        self.model.eval()

        # load dlib model for face detection and landmark  for face cropping
        dlib_landmark_model = os.path.join(baseFolder, 'models/shape_predictor_68_face_landmarks.dat')
        self.face_regressor = dlib.shape_predictor(dlib_landmark_model)
        self.face_detector = dlib.get_frontal_face_detector()

        self.tri = sio.loadmat(os.path.join(baseFolder, 'visualize/tri.mat'))['tri']
        self.transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])


    def forward(self, img_ori, saveFolder, imgName):
        rects = self.face_detector(img_ori, 1)
        if len(rects) == 0:
            return
        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = saveFolder

        rect = rects[0]
        # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
        # - use landmark for cropping
        pts = self.face_regressor(img_ori, rect).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]).T  # detected landmarks, should record this
        roi_box = parse_roi_box_from_landmark(pts)
        img = crop_img(img_ori, roi_box)

        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            if self.mode == 'gpu':
                input = input.cuda()
            param = self.model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # 68 pts
        pts68 = predict_68pts(param, roi_box)

        #pts_res.append(pts68)
        pts_res.append(pts)
        P, pose = parse_pose(param)
        Ps.append(P)
        poses.append(pose)

        # dense face 3d vertices
        vertices = predict_dense(param, roi_box)
        vertices_lst.append(vertices)

        # save projected 3D points
        wfp = '{}_projected.txt'.format(os.path.join(saveFolder, imgName))
        np.savetxt(wfp, pts68, fmt='%.3f')
        #print('Save 68 3d landmarks to {}'.format(wfp))

        wfp = '{}_detected.txt'.format(os.path.join(saveFolder, imgName))
        np.savetxt(wfp, pts, fmt='%.3f')
        #print('Save 68 3d landmarks to {}'.format(wfp))

        # save obj file
        wfp = '{}.obj'.format(os.path.join(saveFolder, imgName))
        colors = get_colors(img_ori, vertices)
        write_obj_with_colors(wfp, vertices, self.tri, colors)
        #print('Dump obj with sampled texture to {}'.format(wfp))

        wfp = os.path.join(saveFolder, imgName) + '_depth.png'
        depths_img = cget_depths_image(img_ori, vertices_lst, self.tri - 1)  # cython version
        cv2.imwrite(wfp, depths_img)
        #print('Dump to {}'.format(wfp))

        draw_landmarks(img_ori, pts_res, wfp=os.path.join(saveFolder, imgName)+'_3DDFA.png', show_flg=False)
