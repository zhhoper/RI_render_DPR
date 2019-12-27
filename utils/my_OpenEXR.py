'''
    define some simple functions to save and load .exr images
'''
import OpenEXR
import Imath
import array
import numpy as np

def saveEXR_RGB(saveName, img):
    '''
        save images as exr
    '''
    rows = img.shape[0]
    cols = img.shape[1]
    if img.ndim == 3:
        # RGB images
        dataR = array.array('f', img[:,:,0].ravel()).tostring()
        dataG = array.array('f', img[:,:,1].ravel()).tostring()
        dataB = array.array('f', img[:,:,2].ravel()).tostring()
    elif img.ndim == 2:
        # gray scale images
        dataR = array.array('f', img.ravel()).tostring()
        dataG = array.array('f', img.ravel()).tostring()
        dataB = array.array('f', img.ravel()).tostring()

    exr = OpenEXR.OutputFile(saveName, OpenEXR.Header(cols,rows))
    exr.writePixels({'R': dataR, 'G': dataG, 'B': dataB})

def loadEXR_RGB(srcName):
    '''
        load exr images as RGB file
    '''
    file = OpenEXR.InputFile(srcName)
    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]
    R = np.array(R)
    G = np.array(G)
    B = np.array(B)

    img = np.zeros((sz[1], sz[0], 3)) # RGB 3 channel
    img[:,:,0] = np.reshape(R, (sz[1], sz[0]))
    img[:,:,1] = np.reshape(G, (sz[1], sz[0]))
    img[:,:,2] = np.reshape(B, (sz[1], sz[0]))
    return img
