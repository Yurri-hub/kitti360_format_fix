import tqdm
import os

# pcd_root = '/home/public/KITTI360/scene0/0/'
# for pcd in tqdm.tqdm(os.listdir(pcd_root)):
#     num = str(int(pcd[:10]))
#     cmd = 'cp ' + pcd_root + pcd + ' /home/public/KITTI360/data_3d_pcd/2013_05_28_drive_0000_sync/velodyne_points/data/' + num + '.pcd'
#     os.system(cmd)

png_root = '/home/public/KITTI360/data_2d_raw/2013_05_28_drive_0000_sync/image_01/data_rect/'
for png in tqdm.tqdm(os.listdir(png_root)):
    num = str(int(png[:10]))
    cmd = 'cp ' + png_root + png + ' /home/public/KITTI360/scene0/image/1/' + num + '.png'
    os.system(cmd)

png_root = '/home/public/KITTI360/data_2d_raw/2013_05_28_drive_0000_sync/image_02/data_rgb/'
for png in tqdm.tqdm(os.listdir(png_root)):
    num = str(int(png[:10]))
    cmd = 'cp ' + png_root + png + ' /home/public/KITTI360/scene0/image/2/' + num + '.png'
    os.system(cmd)

png_root = '/home/public/KITTI360/data_2d_raw/2013_05_28_drive_0000_sync/image_03/data_rgb/'
for png in tqdm.tqdm(os.listdir(png_root)):
    num = str(int(png[:10]))
    cmd = 'cp ' + png_root + png + ' /home/public/KITTI360/scene0/image/3/' + num + '.png'
    os.system(cmd)