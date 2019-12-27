'''
    script to generate node
'''
import numpy as np
import os

numPoints = 50

def loadNode(fileName):
    '''
        load node file
    '''
    count = 0
    node_list = []
    with open(fileName) as f:
        for line in f:
            tmp = line.strip().split()
            if count != 0 and tmp[0] != '#':
                node_list.append([float(item) for item in tmp[1:]])
            count += 1
    return node_list

def get_node(albedo_landmark, face_landmark, imgWidth, imgHeight, saveFolder):
    '''
        create a node file for arap

        Given landmark of albedo and face, this function 
        create a triangle and a correspondence file,
        triangle file: triangulation of albedo landmarks
        correspondence file: correspondence of key points
    '''

    # get boundary points

    # create boundary points
    boundary_points = []
    x_range = np.linspace(0, imgWidth-1, numPoints)
    y_range = np.linspace(0, imgHeight-1, numPoints)
    total_num = 4*numPoints-4  + albedo_landmark.shape[0]

    # save for albedo_landmarks.node
    saveName = os.path.join(saveFolder, 'albedo_landmarks.node')
    saveName_1 = os.path.join(saveFolder, 'correspondence.txt')
    fid = open(saveName, 'w')
    fid_1 = open(saveName_1, 'w')
    print('%d %d %d %d' % (total_num, 2, 1, 1), file=fid)
    count = 1
    for (i, point) in enumerate(albedo_landmark):
        print('%d %0.8f %0.8f %d %d' % (count, point[0], point[1], 1, 0), file=fid)
        print('%d %0.8f %0.8f' % (count-1, face_landmark[i][0], face_landmark[i][1]), file=fid_1)
        count += 1
    # up boundary
    for i in range(numPoints):
        print('%d %0.4f %0.4f %d %d' % (count, x_range[i], 0, 1, 1), file=fid)
        print('%d %0.8f %0.8f' % (count-1, x_range[i], 0), file=fid_1)
        count += 1
    # bottom boundary
    for i in range(numPoints):
        print('%d %0.4f %0.4f %d %d' % (count, x_range[i], imgHeight-1, 1, 1), file=fid)
        print('%d %0.8f %0.8f' % (count-1, x_range[i], imgHeight-1), file=fid_1)
        count += 1
    # left boundary
    for i in range(1, numPoints-1):
        print('%d %0.4f %0.4f %d %d' % (count, 0, y_range[i], 1, 1), file=fid)
        print('%d %0.8f %0.8f' % (count-1, 0, y_range[i]), file=fid_1)
        count += 1
    # right boundary
    for i in range(1, numPoints-1):
        print('%d %0.4f %0.4f %d %d' % (count, imgWidth-1, y_range[i], 1, 1), file=fid)
        print('%d %0.8f %0.8f' % (count-1, imgWidth-1, y_range[i]), file=fid_1)
        count += 1
    fid.close()
    fid_1.close()
    cmd_2 = '../useful_code/triangle_berkeley/triangle -q30 ' \
            + os.path.join(saveFolder, 'albedo_landmarks')
    os.system(cmd_2)

    # create triangle
    node_list =  loadNode(os.path.join(saveFolder, 'albedo_landmarks.1.node'))
    face_list =  loadNode(os.path.join(saveFolder, 'albedo_landmarks.1.ele'))
    print(len(node_list))
    print(len(face_list))
    # create file for arap
    saveName = os.path.join(saveFolder, 'triangle.txt')
    fid = open(saveName, 'w')
    for item in node_list:
        print('v {:0.8f} {:0.8f}'.format(item[0], item[1]), file=fid)
    for item in face_list:
        print('f {:d} {:d} {:d}'.format(int(item[0]), int(item[1]), int(item[2])),file=fid)
    fid.close()
    return
