from detect_landmark import *
import numpy as np
import cv2

detect_landmark = detect_landmark()


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
        if os.path.exists(os.path.join(subFolder, 'albedo_3DFFA.png')):
            print('existing')
            continue
        img = cv2.imread(os.path.join(subFolder, 'albedo.png'))
        albedo_landmark = detect_landmark.detect(img)
        if albedo_landmark is None:
            continue
        else:
            detect_landmark.save_landmark(albedo_landmark, 
                    os.path.join(subFolder, 'albedo_detected.txt'))
            detect_landmark.draw_landmark(albedo_landmark, img, 
                    os.path.join(subFolder, 'albedo_3DDFA.png'))
    else:
        print('no file {}'.format(imgName))
