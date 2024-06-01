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

train_areas = []
test_areas = ['parking1', 'parking2', 'parking3', 'parking4']
is_test = False
scale = 512.0/2896.0
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


def save_labels(dir_in, area_names, dir_out):
    path_colors_label = os.path.join(dir_out, 'assets', 'scan3r160pallete.npy')
    with open(path_colors_label, 'rb') as f:
        semcolors = np.load(f)
    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_label_in = os.path.join(dir_in, area, 'pano', 'semantic')
        dir_label_out = os.path.join(dir_out, area, 'pano', 'semantic')
        dir_label_pretty_out = os.path.join(dir_out, area, 'pano', 'semantic_pretty')
        if not os.path.isdir(dir_label_out):
            os.makedirs(dir_label_out)
        if not os.path.isdir(dir_label_pretty_out):
            os.makedirs(dir_label_pretty_out)

        path_label_in = os.path.join(dir_label_in, name + '_semantic.npy')
        path_label_out = os.path.join(dir_label_out, name + '_semantic.npy')
        path_label_pretty_out = os.path.join(dir_label_pretty_out, name + '_semantic_pretty.png')

        if scale == 1.0:
            with open(path_label_in, 'rb') as f:
                label = np.load(path_label_in)
            shutil.copyfile(path_label_in, path_label_out)
        else:
            with open(path_label_in, 'rb') as f:
                label = np.load(path_label_in)
            label = cv2.resize(label, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            assert label.shape[:2] == hw
            with open(path_label_out, 'wb') as f:
                np.save(f, label)
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
    save_labels(dir_in, area_names, dir_out)


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