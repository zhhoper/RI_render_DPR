import numpy as np
import cv2
import math

def pad_squareVec(vec):
    '''
        pad vector so it has n^2 length
    '''
    if len(vec.shape) != 1:
        raise ValueError('shape of vec should be 1')
    numPixel = vec.shape[0]
    root_num = int(math.ceil(np.sqrt(numPixel)))
    totalPixel = root_num**2
    newVec = np.zeros(totalPixel)
    newVec[0:numPixel] = vec
    return newVec, root_num, numPixel


def textureSampling(textureImg, uv_coordinate):
    '''
        textureImg: is an texture map
        uv_coordinate: nx2, uv coordinate of n points
        return texture for each point
    '''
    # convert uv_coordinate to image coordinate
    imgHeight = textureImg.shape[0]
    imgWidth = textureImg.shape[1]
    channel = textureImg.shape[2]
    tmp_x = uv_coordinate[:,0]*imgWidth
    tmp_y = uv_coordinate[:,1]*imgHeight

    # opencv cannot deal with an image with one side larger than 32767
    # we need to reshape tmp_x and tmp_y
    tmp_x, root_num_x, numPixel_x = pad_squareVec(tmp_x)
    tmp_y, root_num_y, numPixel_y = pad_squareVec(tmp_y)
    if root_num_x != root_num_y or numPixel_x != numPixel_y:
        raise ValueError('x and y should be the same dimension')

    tmp_x = np.reshape(tmp_x, (root_num_x, -1))
    tmp_y = np.reshape(tmp_y, (root_num_y, -1))

    texture = cv2.remap(textureImg.astype(np.float32), tmp_x.astype(np.float32),
        tmp_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    texture = np.reshape(texture, (-1, channel))
    return texture[0:numPixel_x,:]

def getNormal(normal, modelToCameraMatrix):
    '''
        get normal for each pixel
        normal: nx3 matrix, normalized normal
        modelToCameraMatrix: 4x4 transformation matrix
    '''
    #h_normal = np.concatenate((normal, np.ones((normal.shape[0],1))), axis=-1)
    v_normal = np.matmul(normal, np.linalg.inv(modelToCameraMatrix[0:3,0:3]))
    #v_normal = v_normal[:,0:3]
    v_normal = v_normal / np.tile(np.linalg.norm(v_normal, keepdims=True, axis=1), (1, 3))
    return v_normal
