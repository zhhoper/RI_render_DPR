'''
    python2, use pytorch_conda environment
'''
import numpy as np
import cv2
from generate_node import *
import sys
import os


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
    imgName = item.split('.')[0]
    subFolder = os.path.join(savePath, imgName, 'render')
    if os.path.exists(subFolder):
        img = cv2.imread(os.path.join(subFolder, 'albedo.png'))
        face_landmark = np.loadtxt(os.path.join(savePath, imgName, imgName + '_detected.txt')).T
        if os.path.isfile(os.path.join(subFolder, 'albedo_detected.txt')):
            if os.path.exists(os.path.join(subFolder, 'arap.obj')):
                continue
            albedo_landmark = np.loadtxt(os.path.join(subFolder, 'albedo_detected.txt')).T
            get_node(albedo_landmark, face_landmark, img.shape[1], img.shape[0], subFolder)
            # arap
            triangle_path = os.path.join(subFolder, 'triangle.txt')
            correspondence_path = os.path.join(subFolder, 'correspondence.txt')
            saveName = os.path.join(subFolder, 'arap.obj')
            
            cmd = '../utils/libigl_arap/my_arap ' + \
                    triangle_path + ' ' + correspondence_path + ' ' \
                    + saveName + ' ' + str(img.shape[1]) + ' ' + str(img.shape[0])
            os.system(cmd)
        else:
            print('no file {}'.format(imgName))
