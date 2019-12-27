'''
    warp image based on arap results
    Basic ideas:
        First warp UV map,
        all the other textures are warped by adapting UV map
'''
import numpy as np
from get_pixelValue import *
import sys
sys.path.append('../utils/cython')
import mesh_core_cython
from help_warp import *
import os

imgList = ['normal', 'uv', 'albedo', 'normal', 'vertex_3D', 'label']

def get_warpedImage(UV, imgWidth, imgHeight, dataPath, savePath):
    '''
        based on UV map and source image, warp image to target image
    '''
    #labelPath = os.path.join(dataPath, 'label')
    #saveLabelPath = os.path.join(savePath, 'label')
    #if not os.path.exists(saveLabelPath):
    #    os.makedirs(saveLabelPath)

    for imageType in imgList:
        #print imageType
        if imageType != 'albedo':
            featureImg = np.load(os.path.join(dataPath, imageType + '.npy'))
        else:
            featureImg = cv2.imread(os.path.join(dataPath, imageType + '.png'))
            featureImg = featureImg.astype(np.float)/255.0
        outImg = textureSampling(featureImg, UV)
        outImg = np.reshape(outImg, (imgWidth, imgHeight, -1))
        if imageType == 'normal':
            tmp_img = np.linalg.norm(outImg, axis=2)
            mask = np.logical_and(np.abs(tmp_img-1)<0.5, outImg[:,:,2] <= 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            mask = np.tile(mask[...,None], (1,1,3))
            cv2.imwrite(os.path.join(savePath, 'mask.png'), (mask*255.0).astype(np.uint8))

        if imageType != 'albedo':
            np.save(os.path.join(savePath, imageType + '.npy'), outImg)
        visImg = my_visualizeImages(outImg, imageType)*mask
        cv2.imwrite(os.path.join(savePath, imageType + '.png'), visImg.astype(np.uint8))
