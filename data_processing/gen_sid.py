import getopt
import shutil
import sys
import numpy as np
import os
from PIL import Image
import scipy.io as scio
import cv2
import json
from multiprocessing import Process

from data_processing.utils.rgbd_util import getHHA

train_areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_6']
test_areas = ['area_5a', 'area_5b']
is_test = False
scale = 0.25
hw = (512, 1024)


def save_imgs(dir_in, area_names, dir_out):
    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_img_in = os.path.join(dir_in, area, 'pano', 'rgb')
        dir_img_out = os.path.join(dir_out, area, 'pano', 'rgb')
        if not os.path.isdir(dir_img_out):
            os.makedirs(dir_img_out)

        path_img_in = os.path.join(dir_img_in, name + '_rgb.png')
        path_img_out = os.path.join(dir_img_out, name + '_rgb.png')
        if scale == 1.0:
            shutil.copyfile(path_img_in, path_img_out)
        else:
            img = cv2.imread(path_img_in, cv2.IMREAD_COLOR)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            assert img.shape[:2] == hw
            cv2.imwrite(path_img_out, img)

        print('img', i, area_name)
        if is_test:
            break


def save_normals(dir_in, area_names, dir_out):
    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_nor_in = os.path.join(dir_in, area, 'pano', 'normal')
        dir_nor_out = os.path.join(dir_out, area, 'pano', 'normal')
        if not os.path.isdir(dir_nor_out):
            os.makedirs(dir_nor_out)

        path_nor_in = os.path.join(dir_nor_in, name + '_normals.png')
        path_nor_out = os.path.join(dir_nor_out, name + '_normals.png')
        if scale == 1.0:
            shutil.copyfile(path_nor_in, path_nor_out)
        else:
            nor = cv2.imread(path_nor_in, cv2.IMREAD_COLOR)
            nor = cv2.resize(nor, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            assert nor.shape[:2] == hw
            cv2.imwrite(path_nor_out, nor)

        print('nor', i, area_name)
        if is_test:
            break


def save_depth(dir_in, area_names, dir_out):
    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_dep_in = os.path.join(dir_in, area, 'pano', 'depth')
        dir_dep_out = os.path.join(dir_out, area, 'pano', 'depth')
        if not os.path.isdir(dir_dep_out):
            os.makedirs(dir_dep_out)

        path_dep_in = os.path.join(dir_dep_in, name + '_depth.png')
        path_dep_out = os.path.join(dir_dep_out, name + '_depth.png')
        if scale == 1.0:
            shutil.copyfile(path_dep_in, path_dep_out)
        else:
            dep = cv2.imread(path_dep_in, cv2.IMREAD_UNCHANGED)
            dep = cv2.resize(dep, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            assert dep.shape[:2] == hw
            cv2.imwrite(path_dep_out, dep)

        print('dep', i, area_name)
        if is_test:
            break


def save_hha(dir_in, area_names, dir_out):
    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_dep_in = os.path.join(dir_in, area, 'pano', 'depth')
        dir_hha_out = os.path.join(dir_out, area, 'pano', 'hha')
        if not os.path.isdir(dir_hha_out):
            os.makedirs(dir_hha_out)

        dir_pose_in = os.path.join(dir_in, area, 'pano', 'pose')
        dir_pose_out = os.path.join(dir_out, area, 'pano', 'pose')
        if not os.path.isdir(dir_pose_out):
            os.makedirs(dir_pose_out)
        
        path_pose_in = os.path.join(dir_pose_in, name + '_pose.json')
        path_pose_out = os.path.join(dir_pose_out, name + '_pose.json')

        with open(path_pose_in, 'r', encoding='utf8') as f:
            pose = json.loads(f.read())
        camera_mat = np.array(pose['camera_k_matrix'], dtype=np.float32)

        path_dep_in = os.path.join(dir_dep_in, name + '_depth.png')
        path_hha_out = os.path.join(dir_hha_out, name + '_hha.png')

        depth = cv2.imread(path_dep_in, cv2.IMREAD_UNCHANGED)
        if scale == 1.0:
            shutil.copyfile(path_pose_in, path_pose_out)
        else:
            camera_mat[:2] *= scale # scale camera matrix
            depth = cv2.resize(depth, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            assert depth.shape[:2] == hw
            with open(path_pose_out, 'w', encoding='utf8') as f:
                json.dump({'camera_k_matrix': camera_mat.tolist(), 'camera_uuid': pose['camera_uuid'], 'frame_num': pose['frame_num'], 'room': pose['room']}, f)

        # Depth images are stored as 16-bit PNGs and
        # have a maximum depth of 128m and a sensitivity of 1/512m(65535 is maximum depth).
        # No depth information and invalid depth information are stored as 65535.
        depth = np.where(depth==65535, 0.0, depth/512.0)

        hha = getHHA(camera_mat, depth, depth)  # input depth (m)
        cv2.imwrite(path_hha_out, cv2.cvtColor(hha, cv2.COLOR_RGB2BGR))

        print('hha', i, area_name)
        if is_test:
            break


def save_labels(dir_in, area_names, dir_out):
    path_json_label = os.path.join(dir_out, 'assets', 'semantic_labels.json')
    with open(path_json_label, 'r', encoding='utf8') as f:
        id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
    path_name_label = os.path.join(dir_out, 'assets', 'name2label.json')
    with open(path_name_label, 'r', encoding='utf8') as f:
        name2id = json.load(f)
    path_colors_label = os.path.join(dir_out, 'assets', 'colors.npy')
    with open(path_colors_label, 'rb') as f:
        semcolors = np.load(f)
    id2label = np.array([name2id[name] for name in id2name], np.uint8)
    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_label_in = os.path.join(dir_in, area, 'pano', 'semantic')
        dir_label_out = os.path.join(dir_out, area, 'pano', 'semantic')
        dir_label_pretty_out = os.path.join(dir_out, area, 'pano', 'semantic_pretty')
        if not os.path.isdir(dir_label_out):
            os.makedirs(dir_label_out)
        if not os.path.isdir(dir_label_pretty_out):
            os.makedirs(dir_label_pretty_out)

        path_label_in = os.path.join(dir_label_in, name + '_semantic.png')
        path_label_out = os.path.join(dir_label_out, name + '_semantic.png')
        path_label_pretty_out = os.path.join(dir_label_pretty_out, name + '_semantic_pretty.png')

        if scale == 1.0:
            label = cv2.imread(path_label_in, cv2.IMREAD_COLOR)
            shutil.copyfile(path_label_in, path_label_out)
        else:
            label = cv2.imread(path_label_in, cv2.IMREAD_COLOR)
            label = cv2.resize(label, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            assert label.shape[:2] == hw
            cv2.imwrite(path_label_out, label)

        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB).astype(np.int32)
        unk = (label[..., 0] != 0)
        label = id2label[label[..., 1] * 256 + label[..., 2]]
        label[unk] = 0
        cv2.imwrite(path_label_pretty_out, cv2.cvtColor(semcolors[label], cv2.COLOR_RGB2BGR))

        print('label', i, area_name)
        if is_test:
            break


def get_names(dir_in, area):
    dir_img = os.path.join(dir_in, area, 'pano', 'rgb')
    name_exts = os.listdir(dir_img)
    names = []
    for name_ext in name_exts:
        name = name_ext.split('.')[0].strip()
        if len(name) != 0:
            name = '_'.join(name.split('_')[:-1])
            names.append(area + ' ' + name)
    return names


def load_list(dir_in, dir_out):
    train_list = []
    for area in train_areas:
        train_list += get_names(dir_in, area)

    test_list = []
    for area in test_areas:
        test_list += get_names(dir_in, area)

    # def write_txt(path_list, list_ids):
    #     with open(path_list, 'w') as f_list:
    #         f_list.write('\n'.join(list_ids))

    # path_list = os.path.join(dir_out, 'train.txt')
    # write_txt(path_list, train_list)

    # path_list = os.path.join(dir_out, 'test.txt')
    # write_txt(path_list, test_list)

    return train_list, test_list


def main(dir_in, dir_out, cpus):
    train_list, test_list = load_list(dir_in, dir_out)
    area_names = train_list + test_list
    # print('area_names', len(area_names))
    save_imgs(dir_in, area_names, dir_out)
    save_normals(dir_in, area_names, dir_out)
    save_depth(dir_in, area_names, dir_out)
    save_labels(dir_in, area_names, dir_out)

    len_sub = len(area_names) // cpus
    chunks_area_names = [area_names[i:i + len_sub] for i in range(0, len(area_names), len_sub)]
    processes = []
    for chunk in chunks_area_names:
        print('chunk', len(chunk))
        p = Process(target=save_hha, args=(dir_in, chunk, dir_out))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # save_hha(dir_in, area_names, dir_out)


if __name__ == '__main__':
    input_dir = ""
    output_dir = ""
    cpu_num = 24
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:c:", ["input_meta_dir=", "output_dir=", "cpus="])
    except getopt.GetoptError:
        print('gen_sid.py -i <input_meta_dir> -o <output_dir> -c <cpus>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('gen_sid.py -i <input_meta_dir> -o <output_dir>')
            sys.exit()
        elif opt in ("-i", "--input_meta_dir"):
            input_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-c", "--cpus"):
            cpu_num = int(arg)

    main(input_dir, output_dir, cpu_num)