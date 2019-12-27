'''
    define a function to deal with one image
'''
import numpy as np
import cv2
from get_pixelValue import *
from my_objLoader_meshlab import *
from collections import OrderedDict
from math import tan
import os

def loadObj(objName):
    '''
        load obj file
    '''
    obj = OBJ(objName, swapyz=False)
    return obj

def loadTexture_simple(obj, albedoName, labelName):
    '''
        load texture data
        obj: instance of obj class, 
            contains 3D position of points, normals, faces and so on
        albedoName: file name for albedo
        return: ordered dictionary containing all textures
    '''
    texture = OrderedDict()
    texture['vertex_3D'] = obj.vertex
    texture['normal'] = obj.normals
    texture['uv'] = obj.texcoords
    albedo = 1.0*cv2.imread(albedoName)/255.0
    albedo = cv2.flip(albedo, 0)
    texture['albedo'] = textureSampling(albedo, texture['uv'])
    # --------------------------------------------------------
    label = 1.0*cv2.imread(labelName, 0)
    label = cv2.flip(label, 0)
    #texture['label']  = textureSampling(label[...,None], texture['uv'])
    # deal with label in another way
    allLabels = np.unique(label)
    numLabels = allLabels.shape[0]
    for item in range(1, numLabels):
        tmpInd = (label== allLabels[item])[...,None]
        texture['label_{:01d}'.format(int(item))] = \
            textureSampling(tmpInd.astype(np.float), texture['uv'])
    #----------------------------------------------------------
    #labelVis = 1.0*cv2.imread(labelVisName)/255.0
    #labelVis = cv2.flip(labelVis,0)
    #texture['labelVis'] = textureSampling(labelVis, texture['uv'])
    return texture
def loadTexture(obj, albedoName, labelName, labelVisName):
    '''
        load texture data
        obj: instance of obj class, 
            contains 3D position of points, normals, faces and so on
        albedoName: file name for albedo
        return: ordered dictionary containing all textures
    '''
    texture = OrderedDict()
    texture['vertex_3D'] = obj.vertex
    texture['normal'] = obj.normals
    texture['uv'] = obj.texcoords
    albedo = 1.0*cv2.imread(albedoName)/255.0
    albedo = cv2.flip(albedo, 0)
    texture['albedo'] = textureSampling(albedo, texture['uv'])
    # --------------------------------------------------------
    label = 1.0*cv2.imread(labelName, 0)
    label = cv2.flip(label, 0)
    #texture['label']  = textureSampling(label[...,None], texture['uv'])
    # deal with label in another way
    allLabels = np.unique(label)
    numLabels = allLabels.shape[0]
    for item in range(1, numLabels):
        tmpInd = (label== allLabels[item])[...,None]
        texture['label_{:01d}'.format(int(item))] = \
            textureSampling(tmpInd.astype(np.float), texture['uv'])
    #----------------------------------------------------------
    labelVis = 1.0*cv2.imread(labelVisName)/255.0
    labelVis = cv2.flip(labelVis,0)
    texture['labelVis'] = textureSampling(labelVis, texture['uv'])
    return texture

def loadTexture2D(obj, albedoName, normalName):
    '''
        load texture data
        obj: instance of obj class, 
            contains 3D position of points, normals, faces and so on
        albedoName: file name for albedo
        return: ordered dictionary containing all textures
    '''
    texture = OrderedDict()
    texture['vertex_3D'] = obj.vertex
    albedo = 1.0*cv2.imread(albedoName)/255.0
    #albedo = cv2.flip(albedo, 0)
    texture['albedo'] = textureSampling(albedo, texture['uv'])
    normal = 1.0*cv2.imread(normalName)
    #normal = cv2.flip(normal, 0)
    texture['normal_2D'] = textureSampling(normal, texture['uv'])
    return texture
