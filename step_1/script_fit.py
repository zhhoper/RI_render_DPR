from fit_3DDFA import *
import cv2
import os

faceList = []
with open('../data.list') as f:
    for line in f:
        faceList.append(line.strip())
imgPath = '../data/'
savePath = '../result'
if not os.path.exists(savePath):
    os.makedirs(savePath)

fit_3DMM = fit_3DDFA('gpu')
for item in faceList:
    imgName = item.split('.')[0]
    subFolder = os.path.join(savePath, imgName)
    if not os.path.exists(subFolder):
        os.makedirs(subFolder)
    img = cv2.imread(os.path.join(imgPath, item))
    fit_3DMM.forward(img, subFolder, item.split('.')[0])
