'''
    load obj file for opengl further rendering
'''
import numpy as np
import cv2


def MTL(filename):
    # this is from other's code, adapted it to return
    # image for our use
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise Exception("mtl file doesn't start with newmtl stmt")
        #elif values[0] == 'map_Kd':
        #    # load the texture referred to by this declaration
        #    mtl[values[0]] = values[1]
        #    #surf = pygame.image.load(mtl['map_Kd'])
        #    image = cv2.imread(mtl['map_Kd'])
        #    image = cv2.flip(image, 0 )
        #    ix, iy, _ = image.shape
        else:
            mtl[values[0]] = [tmp_v for tmp_v in map(float, values[1:])]
    return contents

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertex = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        # if id is not aligned for
        # texture, normal, use this
        self.faces_all = []

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                # load vertex coordinate
                v = [tmp_v for tmp_v in map(float, values[1:4])]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertex.append(v)
            elif values[0] == 'vn':
                v = [tmp_v for tmp_v in map(float, values[1:4])]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append([tmp_vt for tmp_vt in map(float, values[1:3])])
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                # self.images to store the UV map
                #self.mtl = MTL(values[1])
                pass
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0])-1)
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1])-1)
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2])-1)
                    else:
                        norms.append(0)
                t = face[1]
                face[1] = face[2]
                face[2] = t
                self.faces.append(face)
                self.faces_all.append((face, norms, texcoords, material))

        # NOTE: the faces are not aligned
        #       we need to align them for convenience

        tmp_texcoords = np.zeros((len(self.vertex), 2))
        tmp_normals = np.zeros((len(self.vertex), 3))
        for i in range(len(self.faces_all)):
            p_v = self.faces_all[i][0]
            p_n = self.faces_all[i][1]
            p_t = self.faces_all[i][2]
            for (j, p) in enumerate(p_v):
                tn = self.normals[p_n[j]]
                print(tn)
                print(p)
                tmp_normals[p] = np.array(tn)

                tt = self.texcoords[p_t[j]]
                tmp_texcoords[p] = np.array(tt)

        self.normals = tmp_normals
        self.texcoords = tmp_texcoords

        self.normals = np.array(self.normals, dtype='float32')
        # for meshlabe, z coordinate is pointing inwards, 
        # correct that
        self.normals[:,2] = -1*self.normals[:,2]

        self.vertex = np.array(self.vertex, dtype='float32')
        #self.vertex = np.reshape(self.vertex, [-1])

        self.faces = np.array(self.faces, dtype='uint16')
        #self.faces = np.reshape(self.faces, [-1])

        self.texcoords = np.array(self.texcoords, dtype='float32')
        #self.texcoords = np.reshape(self.texcoords, [-1])
        
        #self.normals = np.reshape(self.normals, [-1])
        #self.images = np.reshape(self.images, [-1])
