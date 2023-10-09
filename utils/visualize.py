import os
import numpy as np
import open3d as o3d
import scipy.io as sio

def get_uni_sphere_xyz(H, W):
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = (i+0.5) / W * 2 * np.pi
    v = ((j+0.5) / H - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    sphere_xyz = np.stack([x, y, z], -1)
    return sphere_xyz

def visualize_pointcloud(rgb_img, depth_img, scale=0.001, crop_ratio=80/512, crop_z_above=1.2,
                         init_trans = np.array([200, 0]), init_rotation=np.array([-np.pi/4, 0, 0]),
                         init_zoom = 4., rotate_pcd=np.array([0, 0, 2 * np.pi]),
                         iterations=480, save_image=False, save_path=None):
    # Project to 3d
    H, W = rgb_img.shape[:2]
    xyz = depth_img[..., None] * get_uni_sphere_xyz(H, W)
    xyzrgb = np.concatenate([xyz, rgb_img], axis=-1)

    # Crop the image and flatten
    if crop_ratio > 0:
        assert crop_ratio < 1
        crop = int(H * crop_ratio)
        xyzrgb = xyzrgb[crop:-crop]
    xyzrgb = xyzrgb.reshape(-1, 6)

    # Crop in 3d
    xyzrgb = xyzrgb[xyzrgb[:,2] <= crop_z_above]

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3] * scale)
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:] / 255.)

    R = pcd.get_rotation_matrix_from_xyz(init_rotation) # rotate in x-direction
    pcd = pcd.rotate(R, center=(0,0,0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=2560, height=1440, left=0, top=0)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    ctr.set_zoom(1./init_zoom)
    ctr.translate(x=init_trans[0], y=init_trans[1])

    source = pcd
    for i in range(iterations):
        R = source.get_rotation_matrix_from_xyz(rotate_pcd / iterations) # rotate in y-direction
        pcd = source.rotate(R, center=(0,0,0))
        
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        if save_image:
            os.makedirs(save_path, exist_ok = True)
            vis.capture_screen_image(f"{save_path}/temp_%04d.jpg" % i)
    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

def set_img_color(colors, background, img, pred, gt, show255=False):
    for i in range(0, len(colors)):
        if i != background:
            img[np.where(pred == i)] = colors[i]
    if show255:
        img[np.where(gt==background)] = 255
    return img

def show_prediction(colors, background, img, pred, gt):
    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, pred, gt)
    final = np.array(im)
    return final

def show_img(colors, background, img, clean, gt, *pds):
    im1 = np.array(img, np.uint8)
    #set_img_color(colors, background, im1, clean, gt)
    final = np.array(im1)
    # the pivot black bar
    pivot = np.zeros((im1.shape[0], 15, 3), dtype=np.uint8)
    for pd in pds:
        im = np.array(img, np.uint8)
        # pd[np.where(gt == 255)] = 255
        set_img_color(colors, background, im, pd, gt)
        final = np.column_stack((final, pivot))
        final = np.column_stack((final, im))

    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, gt, True)
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, im))
    return final

def get_colors(class_num):
    colors = []
    for i in range(class_num):
        colors.append((np.random.random((1,3)) * 255).tolist()[0])

    return colors

def get_ade_colors():
    colors = sio.loadmat('./color150.mat')['colors']
    colors = colors[:,::-1,]
    colors = np.array(colors).astype(int).tolist()
    colors.insert(0,[0,0,0])

    return colors


def print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc, class_names=None, show_no_back=False, no_print=False):
    n = iou.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i+1, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iou[i] * 100))
    mean_IoU = np.nanmean(iou)
    lines.append('----------     %-8s: %.3f , %-8s: %.3f , %-8s: %.3f' % ('mAcc', mean_pixel_acc * 100, 'aAcc', pixel_acc * 100, 'mIoU', mean_IoU * 100))
    # mean_IoU_no_back = np.nanmean(iou[1:])
    # if show_no_back:
    #     lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100, 'mean_IU_no_back', mean_IoU_no_back*100,
    #                                                                                                             'freq_IoU', freq_IoU*100, 'mean_pixel_acc', mean_pixel_acc*100, 'pixel_acc',pixel_acc*100))
    # else:
    #     lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100, 'freq_IoU', freq_IoU*100, 
    #                                                                                                 'mean_pixel_acc', mean_pixel_acc*100, 'pixel_acc',pixel_acc*100))
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line


