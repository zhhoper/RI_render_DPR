'''
    render obj file to get albedo, normal, uv map, and labels
    adapted from the rendering code of 3DDFA
'''

import os
import sys
sys.path.append('../utils/cython')
sys.path.append('../utils')
import numpy as np
import mesh_core_cython
import cv2
from load_texture import *
import time


def crender_colors(vertices, triangles, colors, h, w, c=3, BG=None):
    """ render mesh with colors
    Args:
        vertices: [3, nver]
        triangles: [3, ntri]
        colors: [3, nver]
        h: height
        w: width
        c: channel
        BG: background image
    Returns:
        image: [h, w, c]. rendered image./rendering.
    """

    if BG is None:
        image = -np.ones((h, w, c), dtype=np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG
    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    # to C order
    vertices = vertices.T.astype(np.float32).copy(order='C')
    triangles = triangles.T.astype(np.int32).copy(order='C')
    colors = colors.T.astype(np.float32).copy(order='C')

    mesh_core_cython.render_colors_core(
        image, vertices, triangles,
        colors,
        depth_buffer,
        vertices.shape[0], triangles.shape[0],
        h, w, c
    )
    return image, depth_buffer

def my_visualizeImages(image, image_type):
    '''
        convert image to save as png file for visualizatoin purpose
    '''
    def visualize_1(image):
        '''
            image range to be -1 to 1
        '''
        image = ((image+1.0)/2.0*255.0).astype(np.uint8)
        return image

    def visualize_2(image):
        '''
            image range to be 0 to 1
        '''
        image = (255.0*image).astype(np.uint8)
        return image

    def visualize_3(img):
        '''
            image range to be any value
        '''
        maxValue = np.max(img)
        minValue = np.min(img)
        image = ((img - minValue)/(maxValue-minValue)*255.0).astype(np.uint8)
        return image

    options = {
        'vertex_3D' : visualize_3,
        'normal' : visualize_1,
        'vertex_camera' : visualize_1,
        'albedo' : visualize_2,
        'uv' : visualize_2,
        'normal_2D' : visualize_2,
        'label': visualize_3,
        'labelVis': visualize_2}
    image = options[image_type](image)
    return image

def my_render(imgName, objName, modelFolder, saveFolder):
    '''
        render object and save the rendered resutls to saveFolder
    '''
    obj = loadObj(objName)
    # normalize z-buffer
    min_z = np.min(obj.vertex[:,2])
    max_z = np.max(obj.vertex[:,2])
    obj.vertex[:,2] = (obj.vertex[:,2] - min_z)/(max_z - min_z)

    #texture = loadTexture(obj, os.path.join(modelFolder, 'UV.png'), 
    #        os.path.join(modelFolder, 'label_v4.png'), os.path.join(modelFolder, 'color_label_v4.png'))
    texture = loadTexture_simple(obj, os.path.join(modelFolder, 'UV.png'), 
            os.path.join(modelFolder, 'label_v4.png'))
    ori_image = cv2.imread(imgName)
    imgHeight = ori_image.shape[0]
    imgWidth = ori_image.shape[1]

    # concatenate all texture together for rendering
    numVertices = obj.vertex.shape[0]
    numChannels = 0
    numChannels_list = []
    all_texture = None
    for item in texture.keys():
        if all_texture is None:
            all_texture = texture[item]
        else:
            all_texture = np.concatenate((all_texture, texture[item]), axis=1)
        numChannels += texture[item].shape[1]
        numChannels_list.append(texture[item].shape[1])

    vertex = np.transpose(obj.vertex, (1,0))
    faces = np.transpose(obj.faces,(1,0))
    all_texture = np.transpose(all_texture, (1,0))
    image, depth_buffer = crender_colors(vertex, faces, all_texture, imgHeight, imgWidth, c=numChannels)

    # first get mask based on normal
    start_channel = 0
    for (i, item) in enumerate(texture.keys()):
        if item == 'normal':
            img = image[:,:,start_channel:start_channel+ numChannels_list[i]]
            # get mask according to normal
            # mask out pixels that normals of z has negative value
            mask = np.logical_and(img[:,:,2] < 0, np.logical_and(img[:,:,0] != -1, np.logical_and(img[:,:,1] != -1, img[:,:,2] != -1)))
            # filling holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            mask = mask.astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = np.tile(mask[...,None], (1,1,3))
            cv2.imwrite(os.path.join(saveFolder, 'mask.png'), (mask*255).astype(np.uint8))
            mask = mask.astype(np.float)
        start_channel += numChannels_list[i]


    t_time = time.time()
    # now save the rendered images
    start_channel = 0
    label = np.zeros((image.shape[0], image.shape[1], 14))
    for (i, item) in enumerate(texture.keys()):
        # maximum channel is 3
        img = np.zeros((image.shape[0], image.shape[1], 3))
        img[:,:,0:numChannels_list[i]] = \
            image[:,:,start_channel:start_channel+ numChannels_list[i]]
        start_channel += numChannels_list[i]

        if item.startswith('label') and item != 'labelVis':
            num = int(item.split('_')[1])
            label[:,:,num] = img[:,:,0]
        else:
            saveName = os.path.join(saveFolder, item)
            #print saveName
            # save as exr
            if item == 'normal' or item == 'uv' or item == 'vertex_3D':
                #saveEXR_RGB(saveName + '.npy', img)
                np.save(saveName + '.npy', img)
            # save as png for visualization
            vis_img = my_visualizeImages(img, item)*mask
            cv2.imwrite(saveName + '.png', vis_img.astype(np.uint8))
            if item == 'albedo':
                saveName = os.path.join(saveFolder, 'combine_albedo')
                vis_img = vis_img*0.5 + ori_image*0.5
                cv2.imwrite(saveName + '.png', vis_img.astype(np.uint8))

    # dealing with label
    label_index = np.argmax(label, axis=2)
    label_index = np.tile(label_index[...,None], (1,1,3))
    saveName = os.path.join(saveFolder,'label') 
    # save as exr
    #saveEXR_RGB(saveName + '.exr', label_index)
    np.save(saveName + '.npy', label_index)
    # save as png for visualization
    vis_img = my_visualizeImages(label_index.astype(np.float), 'label')*mask
    cv2.imwrite(saveName + '.png', vis_img.astype(np.uint8))
    print ('time to save file is %s' % (time.time() - t_time))
