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

def get_warpedUV(vertices, triangles, UV, h, w, c=3, BG=None):
    '''
        render UV map
    '''

    if BG is None:
        image = -np.ones((h, w, c), dtype=np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG
    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    # to C order
    vertices = vertices.T.astype(np.float32).copy(order='C')
    triangles = triangles.T.astype(np.int32).copy(order='C')
    UV = UV.T.astype(np.float32).copy(order='C')

    mesh_core_cython.render_colors_core(
        image, vertices, triangles,
        UV,
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
        index = image < 0
        image[index] = 0
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
