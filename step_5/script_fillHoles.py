import sys
sys.path.append('../utils/')
import cv2
from filling_holes import *
import os
import numpy as np
import time

get_normal = extrapolateNormal()

faceList = []
with open('../data.list') as f:
    for line in f:
        faceList.append(line.strip())
imgPath = '../data'
savePath = '../result'
if not os.path.exists(savePath):
    os.makedirs(savePath)

for item in faceList:
    print(item)
    begin_time = time.time()
    imgName = item.split('.')[0]
    subFolder = os.path.join(savePath, imgName, 'warp')
    print(subFolder)

    if os.path.isfile(os.path.join(subFolder, 'mask.png')):
        # load mask
        mask_normal = cv2.imread(os.path.join(subFolder, 'mask.png'))
        mask_normal = cv2.cvtColor(mask_normal, cv2.COLOR_RGB2GRAY)
        mask_normal = mask_normal/255.0
        # excluding ear and neck
        mask_label = cv2.imread(os.path.join(subFolder, 'label.png'))
        mask_label = mask_label[:,:,0]/255.0*13.0
        mask_label = np.round(mask_label)
        mask_leftEar = np.abs(mask_label - 1) < 1e-6
        mask_rightEar = np.abs(mask_label-2)< 1e-6
        mask_neck = np.abs(mask_label-3) < 1e-6
        cv2.imwrite(os.path.join(subFolder, 'leftEar.png'), (mask_leftEar*255.0).astype(np.uint8))
        cv2.imwrite(os.path.join(subFolder, 'rightEar.png'), (mask_rightEar*255.0).astype(np.uint8))
        cv2.imwrite(os.path.join(subFolder, 'neck.png'), (mask_neck*255.0).astype(np.uint8))

        mask_normal = mask_normal*(1-mask_leftEar)*(1-mask_rightEar)*(1-mask_neck)
        mask_normal = np.abs(mask_normal - 1) < 1e-6
        mask_normal = mask_normal.astype(np.uint8)

        # normals on edges are not accurate, remove them
        kernel = np.ones((9,9),np.uint8)
        mask_normal = cv2.erode(mask_normal.astype(np.uint8), kernel, iterations=4)

        #-----------------------------------------------------
        # -- extend face and mask by 40 pixels to avoid artifacts
        #-----------------------------------------------------
        # resize image and normal mask by 0.5 for fast computation
        extend = 40
        row, col = mask_normal.shape
        mask_normal_extend = np.zeros((row+extend*2, col+extend*2))
        mask_normal_extend[extend+1:row+extend+1, extend+1:col+extend+1] = mask_normal.copy()

        ori_mask_normal = mask_normal.astype(np.uint8)
        mask_normal = cv2.resize(mask_normal_extend.astype(np.float), None, fx=0.5, fy=0.5)
        mask_normal = (np.abs(mask_normal - 1)<1e-6).astype(np.uint8)

        #ori_img = loadEXR_RGB(os.path.join(subFolder, 'normal.exr'))
        ori_img = np.load(os.path.join(subFolder, 'normal.npy'))
        img_extend = np.zeros((row+extend*2, col+extend*2, 3))
        img_extend[extend+1:row+extend+1, extend+1:col+extend+1,:] = ori_img.copy()
        img = cv2.resize(img_extend, None, fx=0.5, fy=0.5)
        tmp_img = np.linalg.norm(img, axis=2)
        tmp_img[mask_normal] = 1e-6
        img, mask = get_normal.fill_edgesNormal_withFace(img, mask_normal)
        if img is None:
            print('no file {}'.format(item))
            continue
        inv_mask = 1 - mask
        img[:,:,0] = get_normal.get_equation( inv_mask.astype(np.uint8), img[:,:,0])
        img[:,:,1] = get_normal.get_equation( inv_mask.astype(np.uint8), img[:,:,1])
        img[:,:,2] = get_normal.get_equation( inv_mask.astype(np.uint8), img[:,:,2])

        img = cv2.resize(img, (row+2*extend, col+2*extend))
        tmp_normal = np.tile(ori_mask_normal[...,None], (1,1,3))
        img = (1-tmp_normal)*img[extend+1:row+extend+1, extend+1:col+extend+1:,] + tmp_normal*ori_img
        
        img = img/np.tile(np.linalg.norm(img, axis=2, keepdims=True), (1,1,3))
        #saveEXR_RGB(os.path.join(subFolder, 'full_normal_faceRegion_faceBoundary_extend.exr'), img)
        np.save(os.path.join(subFolder, 'full_normal_faceRegion_faceBoundary_extend.npy'), img)
        img = ((img + 1)/2*255).astype(np.uint8)
        cv2.imwrite(os.path.join(subFolder, 'full_normal_faceRegion_faceBoundary_extend.png'), img)
        print('time used is {}'.format((time.time() - begin_time)))
    else:
        print('no file {}'.format(item))
