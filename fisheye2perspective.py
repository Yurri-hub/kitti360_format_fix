import numpy as np
import os
import yaml
import cv2
# import enum
from tqdm import tqdm
# import scipy.interpolate
# import matplotlib.pyplot as plt
# import pylab

def load_distortion(fisheye_root):
    path = 'calibration/' + fisheye_root + '.yaml'
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        return cfg['distortion_parameters'].values()
    
def load_intrinsics(fisheye_root):
    path = 'calibration/' + fisheye_root + '.yaml'
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        return cfg['projection_parameters'].values()

def load_mirror_parameters(fisheye_root):
    path = 'calibration/' + fisheye_root + '.yaml'
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        return cfg['mirror_parameters'].values()

def readVariable(fid,name,M,N):
    # rewind
    fid.seek(0,0)
    
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success==0:
        return None
    
    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert(len(line) == M*N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat

def K_change(K):
    times = 0.9
    K_out = K.copy()

    K_out[0,2] -= 0
    K_out[1,2] -= 200
    
    K_out[0,0] *= times
    K_out[1,1] *= times
    K_out[0,2] *= times
    K_out[1,2] *= times
    
    return K_out[:3,:3]
    
if __name__=="__main__":
    image_root = 'data_2d_origin'
    fisheye_root = ['image_02', 'image_03']
    fid = open("calibration/perspective.txt",'r')
    for scene in os.listdir(image_root):
        for root in fisheye_root:
            fisheye_path = image_root + '/' + scene + '/' + root + '/data_rgb'
            xi = np.array(list(load_mirror_parameters(root)))
            k1, k2, p1, p2 = np.array(list(load_distortion(root)))
            gamma1, gamma2, u0, v0 = np.array(list(load_intrinsics(root)))
            K = np.array([[gamma1, 0, u0], [0, gamma2, v0], [0, 0, 1]])
            camera_select = 'P_rect_o_' + root[-2:]
            lastrow = np.array([0,0,0,1]).reshape(1,4)
            K_out = np.concatenate((readVariable(fid, camera_select, 3, 4), lastrow))
            K_out = K_change(K_out)
            for i in K_out:
                for j in i:
                    print(j, end=" ")
                print('0.0', end=" ")
            print('0.0 0.0 0.0 1.0')
            print(fisheye_path)
            for image_file in tqdm((os.listdir(fisheye_path))[:386]):
                image_path = fisheye_path + '/' + image_file
                image_path_o = 'data_2d_raw/' + scene + '/' + root + '/data_rgb/' + image_file
                image = cv2.imread(image_path)
                # plt.imshow(image)
                # pylab.show()
                dis_coeff = np.array([k1, k2, p1, p2])
                out = cv2.omnidir.undistortImage(image, K, dis_coeff, xi, cv2.omnidir.RECTIFY_PERSPECTIVE, Knew=K_out[:3,:3], new_size=(1408,376))
                # out_plt = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                # plt.imshow(out_plt)
                # pylab.show()
                cv2.imwrite(image_path_o, out)
                pass
            