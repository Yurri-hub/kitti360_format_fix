import numpy as np
import os
import yaml
import cv2
import enum
from tqdm import tqdm
import scipy.interpolate
import matplotlib.pyplot as plt
import pylab

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
    times = 0.3
    K_out = K.copy()
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
            K_out = K_change(K)
            for i in K_out:
                for j in i:
                    print(j, end=" ")
                print('0.0', end=" ")
            print('0.0 0.0 0.0 1.0')
            # camera_select = 'P_rect_' + root[-2:]
            # lastrow = np.array([0,0,0,1]).reshape(1,4)
            # K = np.concatenate((readVariable(fid, camera_select, 3, 4), lastrow))
            # K = K_change(K)
            for image_file in tqdm(os.listdir(fisheye_path)):
                image_path = fisheye_path + '/' + image_file
                image_path_o = 'data_2d_raw/' + scene + '/' + root + '/data_rgb/' + image_file
                image = cv2.imread(image_path)
                # plt.imshow(image)
                # pylab.show()
                dis_coeff = np.array([k1, k2, p1, p2])
                out = cv2.omnidir.undistortImage(image, K, dis_coeff, xi, cv2.omnidir.RECTIFY_PERSPECTIVE, Knew=K_out, new_size=(500,500))
                # out_plt = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                # plt.imshow(out_plt)
                # pylab.show()
                cv2.imwrite(image_path_o, out)
                pass



class DistortMode(enum.Enum):
    LINEAR = 'linear'
    NEAREST = 'nearest'

def distort_image(img: np.ndarray, cam_intr: np.ndarray, dist_coeff: np.ndarray,
                  mode: DistortMode = DistortMode.LINEAR, crop_output: bool = True,
                  crop_type: str = "corner") -> np.ndarray:
    """Apply fisheye distortion to an image
    Args:
        img (numpy.ndarray): BGR image. Shape: (H, W, 3)
        cam_intr (numpy.ndarray): The camera intrinsics matrix, in pixels: [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
                            Shape: (3, 3)
        dist_coeff (numpy.ndarray): The fisheye distortion coefficients, for OpenCV fisheye module.
                            Shape: (1, 4)
        mode (DistortMode): For distortion, whether to use nearest neighbour or linear interpolation.
                            RGB images = linear, Mask/Surface Normals/Depth = nearest
        crop_output (bool): Whether to crop the output distorted image into a rectangle. The 4 corners of the input
                            image will be mapped to 4 corners of the distorted image for cropping.
        crop_type (str): How to crop.
            "corner": We crop to the corner points of the original image, maintaining FOV at the top edge of image.
            "middle": We take the widest points along the middle of the image (height and width). There will be black
                      pixels on the corners. To counter this, original image has to be higher FOV than the desired output.
    Returns:
        numpy.ndarray: The distorted image, same resolution as input image. Unmapped pixels will be black in color.
    """
    assert cam_intr.shape == (3, 3)
    assert dist_coeff.shape == (4,)

    imshape = img.shape
    if len(imshape) == 3:
        h, w, chan = imshape
    elif len(imshape) == 2:
        h, w = imshape
        chan = 1
    else:
        raise RuntimeError(f'Image has unsupported shape: {imshape}. Valid shapes: (H, W), (H, W, N)')

    imdtype = img.dtype

    # Get array of pixel co-ords
    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)
    img_pts = np.stack((xv, yv), axis=2)  # shape (H, W, 2)
    img_pts = img_pts.reshape((-1, 1, 2)).astype(np.float32)  # shape: (N, 1, 2)

    # Get the mapping from distorted pixels to undistorted pixels
    undistorted_px = cv2.fisheye.undistortPoints(img_pts, cam_intr, dist_coeff)  # shape: (N, 1, 2)
    undistorted_px = cv2.convertPointsToHomogeneous(undistorted_px)  # Shape: (N, 1, 3)
    undistorted_px = np.tensordot(undistorted_px, cam_intr, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
    undistorted_px = cv2.convertPointsFromHomogeneous(undistorted_px)  # Shape: (N, 1, 2)
    undistorted_px = undistorted_px.reshape((h, w, 2))  # Shape: (H, W, 2)
    undistorted_px = np.flip(undistorted_px, axis=2)  # flip x, y coordinates of the points as cv2 is height first

    # Map RGB values from input img using distorted pixel co-ordinates
    if chan == 1:
        img = np.expand_dims(img, 2)
    interpolators = [scipy.interpolate.RegularGridInterpolator((ys, xs), img[:, :, channel], method=mode.value,
                                                               bounds_error=False, fill_value=0)
                     for channel in range(chan)]
    img_dist = np.dstack([interpolator(undistorted_px) for interpolator in interpolators])

    if imdtype == np.uint8:
        # RGB Image
        img_dist = img_dist.round().clip(0, 255).astype(np.uint8)
    elif imdtype == np.uint16:
        # Mask
        img_dist = img_dist.round().clip(0, 65535).astype(np.uint16)
    elif imdtype == np.float16 or imdtype == np.float32 or imdtype == np.float64:
        img_dist = img_dist.astype(imdtype)
    else:
        raise RuntimeError(f'Unsupported dtype for image: {imdtype}')

    if crop_output:
        # Crop rectangle from resulting distorted image
        # Get mapping from undistorted to distorted
        distorted_px = cv2.convertPointsToHomogeneous(img_pts)  # Shape: (N, 1, 3)
        cam_intr_inv = np.linalg.inv(cam_intr)
        distorted_px = np.tensordot(distorted_px, cam_intr_inv, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
        distorted_px = cv2.convertPointsFromHomogeneous(distorted_px)  # Shape: (N, 1, 2)
        distorted_px = cv2.fisheye.distortPoints(distorted_px, cam_intr, dist_coeff)  # shape: (N, 1, 2)
        distorted_px = distorted_px.reshape((h, w, 2))
        if crop_type == "corner":
            # Get the corners of original image. Round values up/down accordingly to avoid invalid pixel selection.
            top_left = np.ceil(distorted_px[0, 0, :]).astype(np.int)
            bottom_right = np.floor(distorted_px[(h - 1), (w - 1), :]).astype(np.int)
            img_dist = img_dist[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
        elif crop_type == "middle":
            # Get the widest point of original image, then get the corners from that.
            width_min = np.ceil(distorted_px[int(h / 2), 0, 0]).astype(np.int32)
            width_max = np.ceil(distorted_px[int(h / 2), -1, 0]).astype(np.int32)
            height_min = np.ceil(distorted_px[0, int(w / 2), 1]).astype(np.int32)
            height_max = np.ceil(distorted_px[-1, int(w / 2), 1]).astype(np.int32)
            img_dist = img_dist[height_min:height_max, width_min:width_max]
        else:
            raise ValueError

    if chan == 1:
        img_dist = img_dist[:, :, 0]

    return img_dist