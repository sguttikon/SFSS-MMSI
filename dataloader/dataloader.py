import cv2
import io
import PIL
import torch
import numpy as np
from torch.utils import data
import random
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize
from dataloader.RGBXDataset import Stanford2d3dPanDataset
from dataloader.RGBXDataset import Structured3dPanDataset
from dataloader.RGBXDataset import Matterport3dPanDataset
from dataloader.RGBXDataset import Ricoh3dPanDataset

def random_mirror(rgb, gt, modal_x1, modal_x2):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x1 = cv2.flip(modal_x1, 1)
        modal_x2 = cv2.flip(modal_x2, 1)

    return rgb, gt, modal_x1, modal_x2

def random_scale(rgb, gt, modal_x1, modal_x2, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x1 = cv2.resize(modal_x1, (sw, sh), interpolation=cv2.INTER_LINEAR)
    modal_x2 = cv2.resize(modal_x2, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x1, modal_x2, scale

class TrainPre(object):
    def __init__(self, config):
        self.config = config
        self.norm_mean = config.norm_mean
        self.norm_std = config.norm_std
        self.img_fill = 0
        self.seg_fill = config.ignore_index

    def __call__(self, rgb, gt, modal_x1, modal_x2):
        rgb, gt, modal_x1, modal_x2 = random_mirror(rgb, gt, modal_x1, modal_x2)
        if self.config.train_scale_array is not None:
            rgb, gt, modal_x1, modal_x2, scale = random_scale(rgb, gt, modal_x1, modal_x2, self.config.train_scale_array)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, self.img_fill)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, self.seg_fill)
        p_modal_x1, _ = random_crop_pad_to_shape(modal_x1, crop_pos, crop_size, self.img_fill)
        p_modal_x2, _ = random_crop_pad_to_shape(modal_x2, crop_pos, crop_size, self.img_fill)

        p_rgb = normalize(p_rgb, self.norm_mean, self.norm_std)
        p_modal_x1 = normalize(p_modal_x1, self.norm_mean, self.norm_std)
        p_modal_x2 = normalize(p_modal_x2, self.norm_mean, self.norm_std)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x1 = p_modal_x1.transpose(2, 0, 1)
        p_modal_x2 = p_modal_x2.transpose(2, 0, 1)
        
        return p_rgb, p_gt, p_modal_x1, p_modal_x2

class ValPre(object):
    def __call__(self, rgb, gt, modal_x1, modal_x2):
        return rgb, gt, modal_x1, modal_x2

def get_train_loader(engine, dataset, config):
    data_setting = {'dataset_name': config.dataset_name,
                    'dataset_path': config.dataset_path,
                    'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    train_preprocess = TrainPre(config)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler

__FOLD__ = {
    '1_train': ['area_1', 'area_2', 'area_3', 'area_4', 'area_6'],
    '1_val': ['area_5a', 'area_5b'],
    '2_train': ['area_1', 'area_3', 'area_5a', 'area_5b', 'area_6'],
    '2_val': ['area_2', 'area_4'],
    '3_train': ['area_2', 'area_4', 'area_5a', 'area_5b'],
    '3_val': ['area_1', 'area_3', 'area_6'],
    'trainval': ['area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6'],
}

def get_train_pipeline(name, config):
    """Segmentation Data Pipeline"""
    if name.lower() == 'stanford2d3d_pan':
        train_pipeline = get_stanford2d3d_pan_pipeline(config)
    elif name.lower() == 'structured3d_pan':
        train_pipeline = get_structured3d_pan_pipeline(config, area=getattr(config, 'area', 'full'),
                                                       lighting=getattr(config, 'lighting', 'rawlight'))
    elif name.lower() == 'matterport3d_pan':
        train_pipeline = get_matterport3d_pan_pipeline(config)
    else:
        raise NotImplementedError
    
    train_loader = data.DataLoader(train_pipeline,
                                   batch_size=config.batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=True,
                                   pin_memory=True)
    train_sampler = None
    return train_loader, train_sampler
    
def get_stanford2d3d_pan_pipeline(config):
    train_preprocess = TrainPre(config)
    data_setting = {
        'dataset_path': config.dataset_path,
        'rgb': config.rgb,
        'ann': config.ann,
        'modality_x': config.modality_x,
        'ignore_index': config.ignore_index,
        'mask_black': ('dual' not in config.backbone and 'trio' not in config.backbone),
        'train_source': config.train_source,
        'eval_source': config.eval_source,
    }
    return Stanford2d3dPanDataset(setting=data_setting, split_name='train', preprocess=train_preprocess)
  
def get_structured3d_pan_pipeline(config, area='full', lighting='rawlight'):
    train_preprocess = TrainPre(config)
    data_setting = {
        'dataset_path': config.dataset_path,
        'area': config.area,
        'lighting': config.lighting,
        'rgb': config.rgb,
        'ann': config.ann,
        'modality_x': config.modality_x,
        'ignore_index': config.ignore_index,
        'mask_black': True,
        'train_source': config.train_source,
        'eval_source': config.eval_source,
        'test_source': config.test_source,
    }
    return Structured3dPanDataset(setting=data_setting, split_name='train', preprocess=train_preprocess)

def get_matterport3d_pan_pipeline(config):
    train_preprocess = TrainPre(config)
    data_setting = {
        'dataset_path': config.dataset_path,
        'rgb': config.rgb,
        'ann': config.ann,
        'modality_x': config.modality_x,
        'ignore_index': config.ignore_index,
        'mask_black': True,
        'train_source': config.train_source,
        'eval_source': config.eval_source,
        'test_source': config.test_source,
    }
    return Matterport3dPanDataset(setting=data_setting, split_name='train', preprocess=train_preprocess)

def get_ricoh3d_pan_pipeline(config):
    train_preprocess = TrainPre(config)
    data_setting = {
        'dataset_path': config.dataset_path,
        'rgb': config.rgb,
        'ann': config.ann,
        'modality_x': config.modality_x,
        'ignore_index': config.ignore_index,
        'mask_black': ('dual' not in config.backbone and 'trio' not in config.backbone),
        'train_source': config.train_source,
        'eval_source': config.eval_source,
    }
    return Ricoh3dPanDataset(setting=data_setting, split_name='train', preprocess=train_preprocess)