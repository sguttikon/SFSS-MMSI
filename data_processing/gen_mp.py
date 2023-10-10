import getopt
import sys
import numpy as np
import os

train_areas = [
    '17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r', '29hnd4uzFmX', '2azQ1b91cZZ', '2n8kARJN3HM',
    '5LpN3gDmAk7', '5ZKStnWn8Zo', '5q7pvUzZiYa', '759xd9YjKW5', '8194nk5LbLH', '82sE5b5pLXE',
    '8WUmhLawc2A', 'ARNzJeq3xxb', 'D7N2EKCX4Sj', 'E9uDoFAP3SH', 'EDJbREhghzL', 'EU6Fwq7SyZv',
    'JF19kD82Mey', 'JmbYfDe2QKZ', 'PX4nDJXEHrG', 'PuKPg4mmafe', 'QUCTc6BB5sX', 'SN83YJsR3w2',
    'TbHJrupSAjP', 'ULsKaCPVFJR', 'V2XKFyX4ASd', 'VVfe2KiqLaN', 'Vt2qJdWjCF2', 'Vvot9Ly1tCj',
    'VzqfbhrpDEA', 'WYY7iVyf5p8', 'XcA2TqTSSAj', 'YFuZgdQ5vWj', 'YVUC4YcDtcY', 'aayBHfsNo7d',
    'ac26ZMwG7aT', 'cV4RVeZvu5T', 'fzynW3qQPVF', 'gTV8FGcVJC9', 'gxdoqLR6rwA', 'gZ6f7yhEvPG',
    'i5noydFURQK', 'mJXqzFtmKg4', 'oLBMNvg9in8', 'p5wJjkQkbXX', 'qoiz87JEwZ2', 'r1Q1Z4BcV1o',
    'r47D5H71a5s', 's8pcmisQ38h', 'sKLMLpTHeUy', 'sT4fr6TAbpF', 'ur6pFq6Qu1A', 'vyrNrziPKCB',
    'Uxmj2M2itWa', 'RPmz2sHmrrY', 'Pm6F8kyY3z2', 'pLe4wQe7qrG', 'JeFG25nYj2p', 'HxpKQynjfin',
    '7y3sRwLe3Va', '2t7WUuJeko7', 'B6ByNegPMKs', 'S9hNv5qa7GM', 'zsNo4HB9uLZ', 'kEZ7cmS4wCh'
]
validation_areas = [
    'UwV83HsGsw3', 'X7HyMhZNoso', 'Z6MFQCViBuw', 'b8cTxDM8gDG', 'e9zR4mvMWw7', 'q9vSo1VnCiC',
    'rPc6DW4iMge', 'rqfALeAoiTq', 'uNb9QFRL6hY', 'wc2JMjhGNzB', 'x8F5xyUWy9e', 'yqstnuAEVhm'
]
test_areas = [
    'VFuaQ6m2Qom', 'VLzqgDo317F', 'ZMojNkEp431', 'jh4fc5c5qoQ', 'jtcxE69GiFV', 'pRbA3pwrgk9',
    'pa4otMbVnkk', 'D7G3Y4RVNrH', 'dhjEzFoUFzH', 'GdvgFV5R1Z5', 'gYvKGZ5eRqb', 'YmJkqBEsHnH'
]

def get_names(dir_in, area):
    dir_img = os.path.join(dir_in, area, 'panorama', 'undistorted_color_images')
    area_ids = os.listdir(dir_img)
    names = []
    for area_id in area_ids:
        name = area_id.split('.')[0].strip()
        if len(name) != 0:
            names.append(area + ' ' + name)
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