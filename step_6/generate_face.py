'''
    including functions used to estimate SH from face and get new face according to ratio image
'''

import cv2
import numpy as np
from utils_SH import *
import scipy.io as io
import random
import cvxpy as cp
import time

# load vertex and compute SH

class get_positiveSH_SDP():
    def __init__(self, verbose = False):
        self.C = io.loadmat('C.mat')['C']
        for i in range(self.C.shape[2]):
            self.C[:,:,i] = (self.C[:,:,i] + self.C[:,:,i].T)/2

        self.c_row, self.c_col, self.c_channel = self.C.shape
        self.verbose = verbose

    def get_SH(self, A, b):
        '''
            A: n x 9, each row is the SH basis of one pixel
            b: n, each element is the target pixel value
            return x: 9-dim SH
            Ax=b s.t. C.*x >>0
        '''
        x = cp.Variable(shape=(self.c_channel))
        T = np.zeros((self.c_row, self.c_col))
        for i in range(self.c_channel):
            T += self.C[:,:,i] * x[i]
        obj = cp.Minimize(cp.sum_squares(A*x-b))
        constraints = [T>>0]
        prob = cp.Problem(obj, constraints)
        begin_time = time.time()
        result = prob.solve()
        if self.verbose:
            print('error is {:f}'.format(result))
            print('time used to solve SDP is {}'.format((time.time() - begin_time)))

        tmpT = np.zeros((self.c_row, self.c_col))
        for i in range(self.c_channel):
            tmpT += self.C[:,:,i] * x.value[i]
        t = np.linalg.eig(tmpT)
        #print t[0]
        tmp = np.dot(A, x.value)
        #print np.min(tmp)
        return x.value

class get_positiveSH_sampling():
    def __init__(self, fileName = 'vertex_122.txt', verbose = False):
        print(fileName)
        tmpVertex = np.loadtxt(fileName)
        self.SH_base = SH_basis_noAtt(tmpVertex).T
        self.verbose = verbose
        self.c_channel = tmpVertex.shape[0]

    def get_SH(self, A, b):
        '''
            A: n x 9, each row is the SH basis of one pixel
            b: n, each element is the target pixel value
            return x: 9-dim SH
            Ax=b s.t. C.*x >>0
        '''
        x = cp.Variable(shape=(self.c_channel))
        A = np.dot(A, self.SH_base)
        obj = cp.Minimize(cp.sum_squares(A*x-b))
        constraints = [x>=0]
        prob = cp.Problem(obj, constraints)
        begin_time = time.time()
        result = prob.solve()
        if self.verbose:
            print('............')
            print('error is {:f}'.format(result))
            print('time used to solve SDP is {}'.format((time.time() - begin_time)))

        return np.dot(self.SH_base, x.value)


class estimateSH_from_faces():
    def __init__(self, method = 'SDP', maxEqus= 1000000, verbose=False):
        # method: which method to use: SDP or sampling method
        #         SDP method or Sampling method
        # maxEqus: maximum number of equations we want to use to get SH
        self.maxEqus = maxEqus  

        if method == 'SDP':
            self.opt_SH = get_positiveSH_SDP()
        elif method == 'Sampling':
            self.opt_SH = get_positiveSH_sampling()

    def get_SH(self, faceImg, skinMask, normal_img):
        '''
            faceImg: M x N matrix
            skinMask: M x N mask, 1 means skin 0 means non-skin
            normalImg: M x N x 3 matrix: x, y z coordinate of normal
        '''
        skinImg = faceImg[skinMask]
        # estimate albedo using the mean of albedo
        albedo = np.sum(skinImg)/np.sum(skinMask)
        #print 'albedo is %f' % albedo
        # get shading
        skinImg = skinImg/albedo
        #print 'skinImg is %f' %  np.max(skinImg)
        baseNormal = np.zeros((np.sum(skinMask), 3))
        for i in range(3):
            tmp = normal_img[:,:,i]
            baseNormal[:,i] = tmp[skinMask]
        b = skinImg

        if len(baseNormal) > self.maxEqus:
            # too many equations, randomly select some to compute SH
            tmp_index = list(range(len(baseNormal)))
            random.shuffle(tmp_index)
            baseNormal = baseNormal[tmp_index[0:self.maxEqus], :]
            b = b[tmp_index[0:self.maxEqus]]

        SH_base = SH_basis(baseNormal)
        return self.opt_SH.get_SH(SH_base, b), skinImg

def generate_face(faceImg, shadingImg, normal_img, sh):
    '''
        generage a new face image under sh using ratio image
        faceImg: original face image
        shadingImg: shading image
        normal_img: normal image
        sh: new sh, supposing sh is 9-dim vector
        return new face image, new shading image and ratio image
    '''
    row, col, _ = normal_img.shape
    new_shadingImg = get_shading(np.reshape(normal_img, (-1, 3)), sh)
    new_shadingImg = np.reshape(new_shadingImg, (row, col))

    LImg = faceImg/255.0
    ratio = new_shadingImg/(shadingImg + 1e-6)
    newFace = LImg*ratio
    ind = newFace > 1.0
    newFace[ind] = 1.0 - 1e-6
    ind = newFace < 0.0
    #print 'negative value is %f' % np.sum(ind)
    newFace[ind] = 1e-6
    newFace = (255.0*newFace).astype(np.uint8)
    #newImg = cv2.cvtColor((255.0*newImg).astype(np.uint8), cv2.COLOR_Lab2BGR)
    return newFace, new_shadingImg, ratio
