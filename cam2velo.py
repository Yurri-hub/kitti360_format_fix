import numpy as np
import os
import tqdm
from scipy.spatial.transform import Rotation as R

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

def write_file(filepath, output):
    length = output[0].shape[0]
    with open(filepath, 'w') as fid:
        for i in range(len(output)):
            op = output[i]
            # str_tmp = str(i) + ' '
            str_tmp = ''
            for j in range(length):
                if j < length - 1:
                    str_tmp += str(op[j]) + ' '
                else: str_tmp += str(op[j]) + '\n'
            fid.write(str_tmp)

pose_root = 'data_poses'
cam2velo = np.array([0.04307104361,-0.08829286498,0.995162929,0.8043914418,-0.999004371,0.007784614041,0.04392796942,0.2993489574,-0.01162548558,-0.9960641394,-0.08786966659,-0.1770225824])
cam2velo = np.reshape(cam2velo, (3,4))
cam2velo = np.concatenate([cam2velo, np.array([[0,0,0,1]])], axis=0)
# print(cam2velo)
velo2cam = np.linalg.inv(cam2velo)
# print(velo2cam)
for scene in tqdm.tqdm(os.listdir(pose_root)):
    cam0pos_file = 'data_poses/' + scene + '/cam0_to_world.txt'
    data = np.loadtxt(cam0pos_file)
    output = []
    for i in range(data.shape[0]):
        RT_tmp = data[i][1:].reshape(4,4)
        RT_tmp = np.matmul(RT_tmp, velo2cam)
        r3 = R.from_matrix(RT_tmp[:3,:3])
        qua = r3.as_quat()
        qua = np.array([qua[3], qua[0], qua[1], qua[2]])
        output.append(np.concatenate([RT_tmp[:3,3], qua], axis=0))
    velo_file = 'data_poses/' + scene + '/velo_to_world.txt'
    write_file(velo_file, output)
    pass
