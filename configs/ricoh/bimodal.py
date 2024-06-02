import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345
C.root = '<sfss_mmsi_path>' # TODO: change this to your own path

remoteip = os.popen('pwd').read()

# Dataset config
"""Dataset Path"""
C.dataset_name = 'Ricoh3D'
C.mapping_name = 'Stanford2D3DS' # TODO: change this to Stanford2D3DS or Structured3D
C.dataset_path = osp.join(C.root, 'datasets', 'Ricoh3D-1K')
C.rgb = 'camera-rgb-1K'
C.ann = 'camera-semantic-1K'
C.modality_x = ['camera-depth-1K'] # TODO: change to 'camera-depth-1K', 'camera-normal-1K', 'camera-hha-1K'
C.train_source = osp.join(C.dataset_path, 'train.txt')
C.eval_source = osp.join(C.dataset_path, 'validation.txt')
C.test_source = osp.join(C.dataset_path, 'test.txt')

"""Image Config"""
C.ignore_index = 255
C.image_height = 512
C.image_width = 512
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'dual_mit_b2' # Remember change the path below.
C.pretrained_model = osp.join(C.root, 'pretrained', 'segformers/mit_b2.pth')
C.decoder = 'DMLPDecoderV2'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'
C.use_dcns = [True, False, False, False]

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 4
C.nepochs = 200
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] # [0.75, 1, 1.25] # 
C.eval_flip = False # True # 
C.eval_crop_size = [512, 1024] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 0
C.checkpoint_step = 1

assert len(C.modality_x) == 1
if C.mapping_name == 'Stanford2D3DS':
    C.fold = 'F1' # TODO: change this to F1, F2 or F3
    C.num_classes = 13
    C.class_names =  ['<UNK>','beam','board','bookcase','ceiling','chair','clutter','column','door','floor','sofa','table','wall','window']
    if C.modality_x[0] == 'camera-depth-1K':
        C.log_dir = osp.abspath(osp.join(C.root, 'workdirs', 'Stanford2D3DS_1024x512/log_' + C.mapping_name + '_' + C.backbone + f'_DMLPDecoderV2_Depth_{C.fold}'))
    elif C.modality_x[0] == 'camera-normal-1K':
        C.log_dir = osp.abspath(osp.join(C.root, 'workdirs', 'Stanford2D3DS_1024x512/log_' + C.mapping_name + '_' + C.backbone + f'_DMLPDecoderV2_Normal_{C.fold}'))
    elif C.modality_x[0] == 'camera-hha-1K':
        C.log_dir = osp.abspath(osp.join(C.root, 'workdirs', 'Stanford2D3DS_1024x512/log_' + C.mapping_name + '_' + C.backbone + f'_DMLPDecoderV2_HHA_{C.fold}'))
    else:
        raise NotImplementedError
elif C.mapping_name == 'Structured3D':
    C.num_classes = 40
    C.class_names =  ['background', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                      'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator',
                      'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub',
                      'bag', 'otherstructure', 'otherfurniture', 'otherprop']
    if C.modality_x[0] == 'camera-depth-1K':
        C.log_dir = osp.abspath(osp.join(C.root, 'workdirs', 'Structured3D_1024x512/log_' + C.mapping_name + '_' + C.backbone + '_DMLPDecoderV2_Depth'))
    elif C.modality_x[0] == 'camera-normal-1K':
        C.log_dir = osp.abspath(osp.join(C.root, 'workdirs', 'Structured3D_1024x512/log_' + C.mapping_name + '_' + C.backbone + '_DMLPDecoderV2_Normal'))
    else:
        raise NotImplementedError
else:
    raise NotImplementedError
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()