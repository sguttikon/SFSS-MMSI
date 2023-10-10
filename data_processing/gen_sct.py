import getopt
import sys
import numpy as np
import os

train_areas = [f'scene_{str(i).zfill(5)}' for i in range(0, 3000)]
validation_areas = [f'scene_{str(i).zfill(5)}' for i in range(3000, 3250)]
test_areas = [f'scene_{str(i).zfill(5)}' for i in range(3250, 3500)]

def get_names(dir_in, area):
    dir_img = os.path.join(dir_in, area, '2D_rendering')
    area_ids = os.listdir(dir_img)
    names = []
    for area_id in area_ids:
        names.append(area + ' ' + area_id)
    return names

def load_list(dir_in, dir_out):
    train_list = []
    for area in train_areas:
        train_list += get_names(dir_in, area)

    validation_list = []
    for area in validation_areas:
        validation_list += get_names(dir_in, area)

    test_list = []
    for area in test_areas:
        test_list += get_names(dir_in, area)

    # def write_txt(path_list, list_ids):
    #     with open(path_list, 'w') as f_list:
    #         f_list.write('\n'.join(list_ids))

    # path_list = os.path.join(dir_out, 'train.txt')
    # write_txt(path_list, train_list)

    # path_list = os.path.join(dir_out, 'validation.txt')
    # write_txt(path_list, validation_list)

    # path_list = os.path.join(dir_out, 'test.txt')
    # write_txt(path_list, test_list)

    return train_list, validation_list, test_list

def main(dir_in, dir_out, cpus):
    train_list, validation_list, test_list = load_list(dir_in, dir_out)
    area_names = train_list + validation_list + test_list

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