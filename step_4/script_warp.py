from my_warp import *
from my_loadMesh import *
import os
import numpy as np
import cv2

faceList = []
with open(os.path.join('../data.list')) as f:
    for line in f:
        faceList.append(line.strip())

imgPath = '../data'
savePath = '../result'

for item in faceList:
    imgName = item.split('.')[0]
    subFolder = os.path.join(savePath, imgName, 'render')
    saveSubFolder = os.path.join(savePath, imgName, 'warp')
    if not os.path.exists(saveSubFolder):
        os.makedirs(saveSubFolder)

    img = cv2.imread(os.path.join(imgPath, item))
    meshName = os.path.join(subFolder, 'arap.obj')
    vertex, faces, tx_coord = load_mesh(meshName)

    imgHeight = img.shape[0]
    imgWidth = img.shape[1]
    vertex = np.transpose(vertex, (1,0))
    faces = np.transpose(faces, (1,0))
    tx_coord = np.transpose(tx_coord, (1,0))

    UV, depth_buffer = get_warpedUV(vertex, faces-1, tx_coord, imgHeight, imgWidth, c=3)
    vis_img = (UV*255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(saveSubFolder, 'UV_warp.png'), vis_img)
    UV = UV[:,:,0:2]
    UV = np.reshape(UV, (-1, 2))

    get_warpedImage(UV, imgWidth, imgHeight, subFolder, saveSubFolder)
    albedo_img = cv2.imread(os.path.join(saveSubFolder, 'albedo.png')).astype(np.float)
    combine_img = 0.5*img.astype(np.float) + 0.5*albedo_img
    cv2.imwrite(os.path.join(saveSubFolder, 'combine_albedo.png'), combine_img.astype(np.uint8))
    
