'''
    load arap.obj
'''
import os
import numpy as np

def load_mesh(meshName):
    '''
        load arap.obj
        return vertices, faces and texture coordinates
    '''

    vertex = []
    faces = []
    texture_coord = []
    with open(meshName) as f:
        for line in f:
            tmp = line.strip().split()
            if len(tmp) == 0:
                pass
            if tmp[0] == 'v':
                vertex.append([float(item) for item in tmp[1:]])
            if tmp[0] == 'vt':
                texture_coord.append([float(item) for item in tmp[1:]])
            if tmp[0] == 'f':
                # index starting from 1
                faces.append([int(item) for item in tmp[1:]])
    return np.array(vertex).astype(np.float32), \
            np.array(faces).astype(np.int), \
            np.array(texture_coord).astype(np.float32)

