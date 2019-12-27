'''
    get normal of a mesh
    based on https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
'''
import numpy as np
eps = 1e-6

def normalize_v3(arr):
    '''
        normalize a numpy array of 3 component vector shape=(n,3)
    '''
    lens = np.sqrt(arr[:,0]**2 + arr[:,1]**2+ arr[:,2]**2)
    lens = lens + eps
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens
    return arr

def get_normal(vertex, faces):
    '''
        compute the normal for each vertex in a mesh
        vertex: nx3 matrix, containing the x,y,z coordinates of a vertex
        faces: nx3 matrix, containing the index of three vertex for a triangle
    '''
    faces = faces - 1 # correct index so it starts from 0

    vertex2face = {}
    vertex_set = set()
    for i in range(len(faces)):
        if faces[i][0] in vertex_set:
            vertex2face[faces[i][0]].append(i)
        else:
            vertex2face[faces[i][0]] = [i]
            vertex_set.add(faces[i][0])

        if faces[i][1] in vertex_set:
            vertex2face[faces[i][1]].append(i)
        else:
            vertex2face[faces[i][1]] = [i]
            vertex_set.add(faces[i][1])

        if faces[i][2] in vertex_set:
            vertex2face[faces[i][2]].append(i)
        else:
            vertex2face[faces[i][2]] = [i]
            vertex_set.add(faces[i][2])
    
    norm = np.zeros(vertex.shape, dtype=vertex.dtype)
    tris = vertex[faces]
    
    # get normal for each triangle
    n = np.cross( tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])

    # normal of a vertex is accumulate of all normals of triangles that contain that vertex
    for i in range(len(vertex)):
        tmp_ind = vertex2face[i]
        tmp_normal = n[tmp_ind]
        norm[i] = np.sum(tmp_normal, axis=0)
    norm = normalize_v3(norm)

    return norm
