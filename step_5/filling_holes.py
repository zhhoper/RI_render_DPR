'''
    filling holes for normals
    using Poisson Equation
'''

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import time
import bisect

class extrapolateNormal():
    def __init__(self):
        # kernel used to get edges
        self.kernel = np.ones((3,3))/9

        # eps to help judge equal or not
        self.eps = 1e-6

    def fill_edgesNormal(self, image, mask):
        '''
            given an image, fill the four boundaries with normals
            first channel:  x pointing right
            second channel: y pointing down
            third channel:  z pointint inward
            NOTE: if there is value in the boundary, we simply replace it
        '''
        numRow, numCol, _ = image.shape

        # normal for four corners 
        x = [0, numCol-1, 0, numRow-1]
        y = [0, 0, numRow-1, numRow-1]
        value = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0]]

        # left boundary
        x.extend([0]*(numRow-2))
        y.extend(range(1, numRow-1))
        value.extend([[-1, 0,0]]*(numRow-2))

        # right boundary
        x.extend([numCol-1]*(numRow-2))
        y.extend(range(1, numRow-1))
        value.extend([[1,0,0]]*(numRow-2))

        # up boundary
        x.extend(range(1, numCol-1))
        y.extend([0]*(numCol-2))
        value.extend([[0, -1, 0]]*(numCol-2))

        # low boundary
        x.extend(range(1,numCol-1))
        y.extend([numRow-1]*(numCol-2))
        value.extend([[0, 1, 0]]*(numCol-2))

        image[y, x, :] = value
        mask[y,x] = 1
        return image, mask

    def fill_edgesNormal_circle(self, image, mask):
        '''
            given an image, fill the four boundaries with normals
            first channel:  x pointing right
            second channel: y pointing down
            third channel:  z pointint inward
            we use a circle to initialize the normal of a circl
        '''
        numRow, numCol, _ = image.shape

        # normal for four corners 
        tmp_x = np.linspace(-1, 1, numCol)
        tmp_y = np.linspace(-1, 1, numRow)
        [normal_x, normal_y] = np.meshgrid(tmp_x, tmp_y)
        mag = np.sqrt(normal_x**2 + normal_y**2)
        normal_x = normal_x / mag
        normal_y = normal_y / mag
        value = np.zeros((numRow, numCol, 3))
        value[:,:,0] = normal_x
        value[:,:,1] = normal_y

        # corner coordinate
        x = [0, numCol-1, 0, numRow-1]
        y = [0, 0, numRow-1, numRow-1]

        # left boundary
        x.extend([0]*(numRow-2))
        y.extend(range(1, numRow-1))

        # right boundary
        x.extend([numCol-1]*(numRow-2))
        y.extend(range(1, numRow-1))

        # up boundary
        x.extend(range(1, numCol-1))
        y.extend([0]*(numCol-2))

        # low boundary
        x.extend(range(1,numCol-1))
        y.extend([numRow-1]*(numCol-2))

        image[y, x, :] = value[y,x,:]
        mask[y,x] = 1
        return image, mask

    def fill_edgesNormal_withFace(self, image, mask):
        '''
            given an image, fill the four boundaries with normals
            first channel:  x pointing right
            second channel: y pointing down
            third channel:  z pointint inward
            we use the mask of face to initialize the normal of a circl
            How to :
            (1) find the boundary of the face region
            (2) get the normal for each pixel on the region, using the center of the image 
                NOTE: if the center of the image is not with in the center of a face, we should ignore this image
            (3) find the angle of each pixel corresponds to the the center of the image
            (4) based on the angle of the image boundary, get the normal of the face
        '''
        numRow, numCol, _ = image.shape
        center_x, center_y = numCol/2, numRow/2
        if mask[int(center_y), int(center_x)] != 1:
            # center of the image is not in the face region
            # ignore this image
            return None, None

        # find the boundary of the face region
        # NOTE: different version of opencv have different way of using findContours
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        ori_img = image.copy()
        cv2.drawContours(ori_img, contours[0], -1, (0,255,0), 3)
        #cv2.imwrite('tmp_contour.png', ori_img)
        # edges = self.get_edges(mask)
        # edges_y, edges_x = np.where(edges)
        edges_x = contours[0][:, 0, 0]
        edges_y = contours[0][:, 0, 1]
        coord_x = edges_x - center_x
        coord_y = edges_y - center_y
        theta = np.arctan2(coord_x, coord_y)
        # get the cooresponding normal
        normal = image[edges_y, edges_x, :]

        ind_theta = np.argsort(theta)
        theta = theta[ind_theta]
        normal = normal[ind_theta]
        z_normal = np.max(normal[:,2])

        # now find the theta for each pixel on the boundary

        # corner coordinate
        x = [0, numCol-1, 0, numRow-1]
        y = [0, 0, numRow-1, numRow-1]

        # left boundary
        x.extend([0]*(numRow-2))
        y.extend(range(1, numRow-1))

        # right boundary
        x.extend([numCol-1]*(numRow-2))
        y.extend(range(1, numRow-1))

        # up boundary
        x.extend(range(1, numCol-1))
        y.extend([0]*(numCol-2))

        # low boundary
        x.extend(range(1,numCol-1))
        y.extend([numRow-1]*(numCol-2))

        theta_boundary = np.arctan2(np.array(x) - center_x, np.array(y)-center_y)
        ind = [bisect.bisect_left(theta, item) for item in theta_boundary]
        ind = np.array(ind)
        t_ind = np.array(ind) == len(theta)
        if np.sum(t_ind) > 0:
            ind[t_ind] = len(theta) - 1

        newNormal = normal[ind]
        newNormal[:,2] = z_normal
        magNormal = np.linalg.norm(newNormal, axis=1)
        newNormal = newNormal/np.tile(magNormal[...,None], (1,3))


        image[y, x, :] = newNormal
        mask[y,x] = 1
        return image, mask

    
    def get_components(self, mask):
        '''
            given mask of an image (0 holes, 1 with values)
            get connected_components that need to file in valus
        '''
        # 8 connected componnets
        # NOTE: index 0 is the background
        components = cv2.connectedComponents( ((mask != 1)*255).astype(np.uint8), connectivity=8)
        return components

    def get_edges(self, mask):
        '''
            given a mask, get edges
            mask binary image, 1: pixels we want to fill values in 
                               0: pixels have values
            an edge is defined as a pixel that has 1 and 0 neighbors and has 0 value in mask
            NOTE: we may include more edges that we needed, but that does not matter, we can 
                  ignore them in the following steps
        '''
        help_edge = cv2.filter2D(mask.astype(np.float32), -1, self.kernel)
        edge = np.logical_and(np.logical_not(np.abs(help_edge) < self.eps),
            np.logical_not(np.abs(help_edge-1) < self.eps))

        # remove pixels that has no valid pixel value
        edge = np.logical_and(edge, mask == 0)
        return edge

    def help_indexValue(self, index_x, index_y, mask, img, b, index):
        '''
            based on index_x and index_y check whether the point is a 
            boundary or not, if it is an boundary, put the bondary value
            in b and delete the value in index_x, index_y
        '''
        tmpValue = np.where(mask[index_x, index_y] == 0)
        #print len(index_x)
        #print len(index_y)
        #print max(tmpValue)
        #print b.shape
        #begin_time = time.time()
        tmpValue = set(tmpValue[0])
        #print 'time convert to list is %s' % (time.time() - begin_time)
        #begin_time = time.time()
        new_x = list(set(range(len(index_x))) - tmpValue)
        #print 'time to get new_x is %s' % (time.time() - begin_time)
        #begin_time = time.time()
        new_y = [index[(index_x[i], index_y[i])] for (i, item) in enumerate(index_y) if not i in tmpValue]
        #print 'time to get new_y is %s' % (time.time() - begin_time)
        #begin_time = time.time()

        for item in tmpValue:
            b[item] += img[index_x[item], index_y[item]]
        #print 'time to get b is %s' % (time.time() - begin_time)
        return new_x, new_y, b

    def get_indexValue(self, x, y, mask, img, index):
        '''
            get index and value for a sparse matrix we need to solve
            x,y: index of the unknown elements
            mask: binary image, 1 -> unknow pixels, 0 -> known pixels
            img: image with size (m x n) containing values
            return  index_x, index_y, value: x, y index and value of A
                    b: value of b
        '''
        index_x = []
        index_y = []
        value = []
        b = np.zeros(x.shape[0])

        shift_x = [0, 0, -1, 1] 
        shift_y = [-1, 1, 0, 0]
        for i in range(4):
            # left, right, up and down
            tmp_x = list(x + shift_x[i])
            tmp_y = list(y + shift_y[i])
        
            tmp_x, tmp_y, b = self.help_indexValue(tmp_x, tmp_y, mask, img, b, index)
            value.extend([1]*len(tmp_x))
            index_x.extend(tmp_x)
            index_y.extend(tmp_y)
        index_x.extend(range(x.shape[0]))
        index_y.extend(range(y.shape[0]))
        value.extend([-4]*x.shape[0])
        return index_x, index_y, value, b


    def get_equation(self, mask, img):
        '''
            mask binary image: 1: pixels we want to fill values in
                               0: pixels we have values
            edge binary image: 1: contains edges that may be used as boundary condition
            img: image with size (m x n)  containing values of boundary conditions
            create a Poisson equation return A and b and index
            Ax = b 
            index indicate the position of the elements of x
        '''
        begin_time = time.time()
        x, y =  np.where(mask)
        tmp_ind = np.concatenate((x[...,None], y[...,None]), axis=1)
        tmp_ind = map(tuple, tmp_ind)
        index = {item:i for (i, item) in enumerate(tmp_ind)}
        help_index = {i:item for (i, item) in enumerate(tmp_ind)}
        numElems = x.shape[0]
        begin_time = time.time()

        # number of variables we need to solve
        index_x, index_y, value, b = self.get_indexValue(x, y, mask, img, index)
        begin_time = time.time()

        A = sp.csr_matrix((value, (index_x, index_y)), shape=(numElems, numElems))
        b = -1*b
        begin_time = time.time()
        results = linalg.cg(A, b) # solve the equation using conjugate gradient descent
        begin_time = time.time()
        # fill the holes
        img[x,y] = results[0]
        return img
