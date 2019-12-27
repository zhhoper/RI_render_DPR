import sys
sys.path.append('../utils/')
import numpy as np
import cv2
from generate_face import *
from utils_SH import *
import scipy.io as io
import os
from utils_normal import *
from utils_shtools import *
from pyshtools.rotate import djpi2, SHRotateRealCoef
from pyshtools.expand import MakeGridDH

# ---------------- create normal for rendering half sphere ------
img_size = 256
x = np.linspace(-1, 1, img_size)
z = np.linspace(1, -1, img_size)
x, z = np.meshgrid(x, z)

mag = np.sqrt(x**2 + z**2)
valid = mag <=1
y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
x = x * valid
y = y * valid
z = z * valid
normal_sphere = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
normal_sphere = np.reshape(normal_sphere, (-1, 3))

# degree of sh
SH_DEGREE = 2
def process_SH(sh):
    '''
        preprocess the lighting:
        normalize the lighting so the DC component will be within 
        range 0.7 to 0.9
    '''
    
    # if sh is a 3x9 vector, conver it to gray
    if sh.shape[0] == 3:
        # RGB2Gray
        coeffs_matrix_R = shtools_sh2matrix(sh[0,:], SH_DEGREE)
        tmp_envMap_R = MakeGridDH(coeffs_matrix_R, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)

        coeffs_matrix_G = shtools_sh2matrix(sh[1,:], SH_DEGREE)
        tmp_envMap_G = MakeGridDH(coeffs_matrix_G, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)

        coeffs_matrix_B = shtools_sh2matrix(sh[2,:], SH_DEGREE)
        tmp_envMap_B = MakeGridDH(coeffs_matrix_B, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)

        tmp_envMap = 0.299*tmp_envMap_R + 0.587*tmp_envMap_G + 0.114*tmp_envMap_B  # from opencv document
        tmp_SH = pyshtools.expand.SHExpandDH(tmp_envMap, sampling=2, lmax_calc=2, norm=4)
        sh = shtools_matrix2vec(tmp_SH)
        
    
    coeffs_matrix = shtools_sh2matrix(sh, SH_DEGREE)
    # random rotate
    dj = djpi2(SH_DEGREE)
    sh_angle = np.random.rand()*2*np.pi
    coeffs_matrix = SHRotateRealCoef(coeffs_matrix, np.array([sh_angle, 0, 0]), dj)

    factor = (np.random.rand()*0.2 + 0.7)/sh[0]
    tmp_envMap = MakeGridDH(coeffs_matrix, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)
    tmp_envMap = tmp_envMap/factor
    tmp_SH = pyshtools.expand.SHExpandDH(tmp_envMap, sampling=2, lmax_calc=2, norm=4)
    
    sh = shtools_matrix2vec(tmp_SH)
    sh = sh[...,None]

    if sh[0] < 0.5 or sh[0] > 1.0:
        # normalize it again if it is too dark or bright
        factor = (np.random.rand()*0.2 + 0.7)/sh[0]
        coeffs_matrix = shtools_sh2matrix(sh, SH_DEGREE)
        tmp_envMap = MakeGridDH(coeffs_matrix, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)
        tmp_envMap = tmp_envMap*factor
        tmp_SH = pyshtools.expand.SHExpandDH(tmp_envMap, sampling=2, lmax_calc=2, norm=4)
    
        sh = shtools_matrix2vec(tmp_SH)
        sh = sh[...,None]
    return sh, factor, sh_angle

def process_SH_simple(sh):
    '''
        preprocess the lighting:
        normalize the lighting so the DC component will be within 
        range 0.7 to 0.9
    '''
    
    # if sh is a 3x9 vector, conver it to gray
    if sh.shape[0] == 3:
        # RGB2Gray
        coeffs_matrix_R = shtools_sh2matrix(sh[0,:], SH_DEGREE)
        tmp_envMap_R = MakeGridDH(coeffs_matrix_R, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)

        coeffs_matrix_G = shtools_sh2matrix(sh[1,:], SH_DEGREE)
        tmp_envMap_G = MakeGridDH(coeffs_matrix_G, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)

        coeffs_matrix_B = shtools_sh2matrix(sh[2,:], SH_DEGREE)
        tmp_envMap_B = MakeGridDH(coeffs_matrix_B, lmax=255, lmax_calc=SH_DEGREE, sampling=2, norm=4)

        tmp_envMap = 0.299*tmp_envMap_R + 0.587*tmp_envMap_G + 0.114*tmp_envMap_B  # from opencv document
        tmp_SH = pyshtools.expand.SHExpandDH(tmp_envMap, sampling=2, lmax_calc=2, norm=4)
        sh = shtools_matrix2vec(tmp_SH)
        
    
    coeffs_matrix = shtools_sh2matrix(sh, SH_DEGREE)
    # random rotate
    dj = djpi2(SH_DEGREE)
    sh_angle = np.random.rand()*2*np.pi
    coeffs_matrix = SHRotateRealCoef(coeffs_matrix, np.array([sh_angle, 0, 0]), dj)

    sh = shtools_matrix2vec(coeffs_matrix)
    factor = (np.random.rand()*0.2 + 0.7)/sh[0]
    sh = sh * factor
    sh = sh[...,None]

    return sh, factor, sh_angle

def process_mask(mask_project, mask_detect):
    '''
        Given projected mask maks_project and detected mask mask_detect
        get the mask of skin region, it is defined as the intersection of
        projected mask and detected mask
    '''
    mask_project = mask_project[:,:,0]/255.0*13.0
    mask_project = np.round(mask_project)
    mask_project = np.abs(mask_project -13.0) < 1e-6

    mask_detect = cv2.resize(mask_detect, (mask_project.shape[0], mask_project.shape[1]))
    mask_detect = np.argmax(mask_detect, axis=2)
    mask_detect = mask_detect == 1 # 1 means skin
    
    mask = np.logical_and(mask_project, mask_detect)
    # remove some boundary pixels since they are not accurate
    kernel = np.ones((11,11), np.uint8)
    mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    mask = mask.astype(np.bool)
    return mask

def process_normal(normal):
    '''
        process loaded normal, normalize normal to be 1
        and align the coordinate of normal with envmap
    '''
    tmp = np.isnan(normal)
    tmp_img = np.linalg.norm(normal, axis=2)
    normal = normal/np.tile(tmp_img[...,None], (1,1,3))
    normal[tmp] = 0

    # transfer the normal to align the coordinate with envmap
    cvtNormal = sh_cvt()
    normal = cvtNormal.cvt2shtools(normal)
    return normal

def script_relighting(numImgs):
    lightPath = 'processed_bip2019'
    lightList = []
    with open('lighting.list')) as f:
        for line in f:
            tmp = line.strip()
            lightList.append(tmp)

    # get list of all images    
    basePath = '../data/'
    faceList = []
    with open('../data.list') as f:
        for line in f:
            faceList.append(line.strip())

    imgPath = '../data'
    dataPath = '../result/'
    relightPath = '../relighting/'
    labelPath = '../face_parsing/'

    for item in faceList:
        print('dealing with {}'.format(item))
        imgName = item.split('.')[0]
        subFolder = os.path.join(dataPath, imgName, 'warp')

        savePath = os.path.join(relightPath, imgName)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        
        #------------------------------------------------------
        # 
        if not os.path.isfile(os.path.join(subFolder, 'full_normal_faceRegion_faceBoundary_extend.npy')):
            print('no normal {}'.format(imgName))
            continue
        if not os.path.isfile(os.path.join(labelPath, imgName + '_detected.mat')):
            print('no label {}'.format(imgName))
            continue
        normal = np.load(os.path.join(subFolder, 'full_normal_faceRegion_faceBoundary_extend.npy'))
        normal = process_normal(normal)

        #------------------------------------------------------
        # load skin mask to compute SH
        # we define the skin region to be the intersection of projected skin mask
        # and detected skin mask by Sifei Liu

        # load projected skin mask
        mask_project = cv2.imread(os.path.join(subFolder, 'label.png'))
        # load detected skin mask
        tmpMat = io.loadmat(os.path.join(labelPath, imgName + '_detected.mat'))
        mask = process_mask(mask_project, tmpMat['res_label'])

        #--------------------------------------------------------
        # compute SH
        img = cv2.imread(os.path.join(imgPath, item))
        row, col, _ = img.shape  # row and col of image
        # denoise using opencv
        img = cv2.fastNlMeansDenoisingColored(img, None, 5, 3, 3, 21)
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        L_img = Lab[:,:,0].copy()
        getSH = estimateSH_from_faces(method = 'SDP', maxEqus = 10000, verbose=True)  # using SDP since it is fast
        sh, skinImg = getSH.get_SH(Lab[:,:,0]/255.0, mask, normal)
        # save computed sh
        np.savetxt(os.path.join(savePath, 'ori_sh.txt'), sh)

        newShading = get_shading(np.reshape(normal, (-1, 3)), sh)
        newShading = np.reshape(newShading, (row, col))
    
        # record negative value as mask
        ind_mask = newShading >0
        cv2.imwrite(os.path.join(savePath, 'relighting_mask.png'), (ind_mask*255.0).astype(np.uint8))

        ind = newShading <= 0
        tmp = newShading.copy()
        tmp[ind] = 100
        # fill negative value with smallest non-zero value
        newShading[ind] = np.min(tmp)

        # save the original shading
        visShading = ( (newShading-np.min(newShading) )/( np.max(newShading) - np.min(newShading) ) )*255.0
        cv2.imwrite(os.path.join(savePath, 'ori_shading.png'), visShading.astype(np.uint8))
        # save the original shading for futher use
        np.save(os.path.join(savePath, 'ori_shading.npy'), newShading)

        # randomly select lighting 
        log_fid = open(os.path.join(os.path.join(savePath, imgName + '_log.txt')), 'w')
        for i in range(numImgs):
            itemName = np.random.choice(lightList)
            lightName = (os.path.join(lightPath, itemName + '.txt'.format(i)))
            sh = np.loadtxt(lightName)
            sh, factor, sh_angle= process_SH_simple(sh)
            np.savetxt(os.path.join(os.path.join(savePath, imgName + '_light_{:02d}.txt'.format(i))), sh)
            print('name {} scale {:0.6f} angle {:0.6f}'.format(itemName, factor, sh_angle), file=log_fid)

            Limg, shading, ratio = generate_face(L_img, newShading, normal, sh)
            Lab[:,:,0] = Limg
            img = cv2.cvtColor(Lab, cv2.COLOR_Lab2BGR)
            cv2.imwrite(os.path.join(savePath, imgName + '_{:02d}.png'.format(i)), img)
            cv2.imwrite(os.path.join(savePath, 'shading_' + imgName + '_{:02d}.png'.format(i)), 
                    (( (shading - np.min(shading))/(np.max(shading) - np.min(shading)))*255.0).astype(np.uint8))

            # rendering half-sphere
            sh = np.squeeze(sh)
            shading = get_shading(normal_sphere, sh)
            ind = shading > 1
            shading[ind] = 1
            shading = (shading *255.0).astype(np.uint8)
            shading = np.reshape(shading, (256, 256))
            shading = shading * valid
            cv2.imwrite(os.path.join(savePath, \
                    'sphere_' + imgName + '_{:02d}.png'.format(i)), shading)
        log_fid.close()
if __name__ == '__main__':
    numImgs = int(sys.argv[1])
    script_relighting(numImgs)
