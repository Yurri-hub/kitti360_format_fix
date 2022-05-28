import numpy as np
import os

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
    with open(filepath, 'w') as fid:
        for i in range(len(output)):
            op = np.reshape(output[i], (16))
            str_tmp = str(i) + ' '
            for j in range(16):
                if j < 15:
                    str_tmp += str(op[j]) + ' '
                else: str_tmp += str(op[j]) + '\n'
            fid.write(str_tmp)

R_rect_00 = [0.999974,-0.007141,-0.000089,0.007141,0.999969,-0.003247,0.000112,0.003247,0.999995]
R_rect_00 = np.reshape(R_rect_00, (3,3))
R_rect_00 = np.concatenate([R_rect_00, [[0,0,0]]], axis=0)
R_rect_00 = np.concatenate([R_rect_00, np.array([[0,0,0,1]]).T], axis=1)
R_rect_01 = [0.999778,-0.012115,0.017222,0.012059,0.999922,0.003351,-0.017261,-0.003143,0.999846]
R_rect_01 = np.reshape(R_rect_01, (3,3))
R_rect_01 = np.concatenate([R_rect_01, [[0,0,0]]], axis=0)
R_rect_01 = np.concatenate([R_rect_01, np.array([[0,0,0,1]]).T], axis=1)
cam2pose_0 = np.array([0.0371783278,-0.0986182135,0.9944306009,1.5752681039,0.9992675562,-0.0053553387,-0.0378902567,0.0043914093,0.0090621821,0.9951109327,0.0983468786,-0.6500000000, 0,0,0,1])
cam2pose_1 = np.array([0.0194000864,-0.1051529641,0.9942668106,1.5977241400,0.9997374956,-0.0100836652,-0.0205732716,0.5981494900,0.0121891942,0.9944049345,0.1049297370,-0.6488433108, 0,0,0,1])
cam2pose_0.resize(4,4)
cam2pose_1.resize(4,4)

cam2pose_2 = np.array([0.9995185086,0.0041276589,-0.0307524527,0.7264036936,-0.0307926666,0.0100608424,-0.9994751579,-0.1499658517,-0.0038160970,0.9999408692,0.0101830998,-1.0686400091, 0,0,0,1])
cam2pose_3 = np.array([-0.9996821702,0.0005703407,-0.0252038325,0.7016842127,-0.0252033830,0.0007820814,0.9996820384,0.7463650950,0.0005898709,0.9999995315,-0.0007674583,-1.0751978255, 0,0,0,1])
cam2pose_2.resize(4,4)
cam2pose_3.resize(4,4)

# https://github.com/autonomousvision/kitti360Scripts/blob/7ecc14eab6fa30e5d2ac71ad37fed2bb4b0b8073/kitti360scripts/helpers/project.py#L35-L41
pose_root = 'data_poses'
for scene in os.listdir(pose_root):
    pos_file = 'data_poses/' + scene + '/poses.txt'
    image_root = 'data_2d_raw/' + scene + '/image_00/data_rect'
    data = np.loadtxt(pos_file)
    output = []
    index = 0
    for i in range(len(os.listdir(image_root))):
        if index < len(data):
            if i == 0:
                output.append(np.concatenate([np.reshape(data[0, 1:], (3, 4)), [[0,0,0,1]]], axis=0))
                continue
            elif i == data[index, 0]:
                output.append(np.concatenate([np.reshape(data[index, 1:], (3, 4)), [[0,0,0,1]]], axis=0))
                index += 1
            else:
                output.append(np.concatenate([np.reshape(data[index-1, 1:], (3, 4)), [[0,0,0,1]]], axis=0))
        else:
            output.append(np.concatenate([np.reshape(data[index-1, 1:], (3, 4)), [[0,0,0,1]]], axis=0))
    output0 = []
    for i in range(len(output)):
        output0.append(np.matmul(np.matmul(output[i], cam2pose_0), np.linalg.inv(R_rect_00)))
    output1 = []
    for i in range(len(output)):
        output1.append(np.matmul(np.matmul(output[i], cam2pose_1), np.linalg.inv(R_rect_01)))
    output2 = []
    for i in range(len(output)):
        output2.append(np.matmul(output[i], cam2pose_2))
    output3 = []
    for i in range(len(output)):
        output3.append(np.matmul(output[i], cam2pose_3))

    pose0_file = 'data_poses/' + scene + '/cam0_to_world.txt'
    pose1_file = 'data_poses/' + scene + '/cam1_to_world.txt'
    pose2_file = 'data_poses/' + scene + '/cam2_to_world.txt'
    pose3_file = 'data_poses/' + scene + '/cam3_to_world.txt'

    write_file(pose0_file, output0)
    write_file(pose1_file, output1)
    write_file(pose2_file, output2)
    write_file(pose3_file, output3)

m = 1