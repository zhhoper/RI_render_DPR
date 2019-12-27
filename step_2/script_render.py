import os
import numpy as np
from getObj_3DDFA import *
from my_render import *
import time

modelFolder = '../data/3DMM/'
triangle_info_path = os.path.join(modelFolder, 'model_info.mat')
objPath = os.path.join(modelFolder, 'BFM_UV.mat')
# change this to absolute path
mtl_path = '/cfarhomes/hzhou/vulcan_data/portraitRelighting/publish/code_prepare/data/3DMM/3DMM_normal.obj.mtl'

getObj = getObj_3DDFA(triangle_info_path, objPath, mtl_path)

imgPath = '../data/'
faceList = []
with open('../data.list') as f:
    for line in f:
        faceList.append(line.strip())

savePath = '../result'
for item in faceList:
    imgName = item.split('.')[0]
    subFolder = os.path.join(savePath, imgName)
    src = os.path.join(subFolder, imgName + '.obj')
    dst = os.path.join(subFolder, imgName + '_new.obj')
    getObj.create_newObj(src, dst)

    objName = os.path.join(subFolder, imgName + '_new.obj')
    imgName = os.path.join(imgPath, item)
    begin_time = time.time()
    saveFolder = os.path.join(subFolder, 'render')
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    my_render(imgName, objName, modelFolder, saveFolder)
    print('dealing with %s used %s seconds' % (imgName, time.time() - begin_time))

