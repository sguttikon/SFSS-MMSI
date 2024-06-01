import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from mmseg.core.evaluation import mean_iou
from dataloader.RGBXDataset import Stanford2d3dPanDataset
from dataloader.RGBXDataset import Structured3dPanDataset
from dataloader.RGBXDataset import Matterport3dPanDataset
from dataloader.RGBXDataset import Ricoh3dPanDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre, __FOLD__
from importlib import import_module

logger = get_logger()

import io

def decode_numpy(extension: str, data: bytes) -> dict | list:
    with io.BytesIO(data) as stream:
        return np.load(stream)

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data[self.rgb]
        label = data[self.ann]
        modal_x1 = data[self.modality_x[0]]
        if len(self.modality_x) == 1:
            modal_x2 = modal_x1
        elif len(self.modality_x) == 2:
            modal_x2 = data[self.modality_x[1]]
        else:
            raise NotImplementedError
        assert set(np.unique(label).tolist()).issubset(self.valid_labels), 'Unknown target label'

        # name = data['fn']
        pred = self.sliding_eval_rgbX(img, modal_x1, modal_x2, self.eval_crop_size, self.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(self.class_num, pred, label)
        results_dict = {'pred': pred, 'label': label, 'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        # if self.save_path is not None:
        #     ensure_dir(self.save_path)
        #     ensure_dir(self.save_path+'_color')

        #     fn = name + '.png'

        #     # save colored result
        #     result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
        #     class_colors = get_class_colors()
        #     palette_list = list(np.array(class_colors).flat)
        #     if len(palette_list) < 768:
        #         palette_list += [0] * (768 - len(palette_list))
        #     result_img.putpalette(palette_list)
        #     result_img.save(os.path.join(self.save_path+'_color', fn))

        #     # save raw result
        #     cv2.imwrite(os.path.join(self.save_path, fn), pred)
        #     logger.info('Save the image ' + fn)

        # if self.show_image:
        #     colors = self.dataset.get_class_colors
        #     image = img
        #     clean = np.zeros(label.shape)
        #     comp_img = show_img(colors, self.background, image, clean,
        #                         label,
        #                         pred)
        #     cv2.imshow('comp_image', comp_img)
        #     cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        # hist = np.zeros((self.class_num, self.class_num))
        # correct = 0
        # labeled = 0
        # count = 0
        # for d in results:
        #     hist += d['hist']
        #     correct += d['correct']
        #     labeled += d['labeled']
        #     count += 1

        # iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        # result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
        #                         self.class_names[1:], show_no_back=False) # ignore background/unknown class
        # return result_line

        preds = []
        gts = []
        for d in results:
            preds.append(d['pred'].astype(np.uint8))
            gts.append(d['label'].astype(np.uint8))
        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        iou_result = mean_iou(results=preds, gt_seg_maps=gts, num_classes=self.class_num,
                              ignore_index=self.ignore_index, nan_to_num=None, label_map=dict(),
                              reduce_zero_label=False)
        return iou_result

def get_eval_pipeline(name, config, split_name='validation'):
    """Segmentation Data Pipeline"""
    if name.lower() == 'stanford2d3d_pan':
        return get_stanford2d3d_pan_pipeline(config, split_name)
    elif name.lower() == 'structured3d_pan':
        return get_structured3d_pan_pipeline(config, split_name)
    elif name.lower() == 'matterport3d_pan':
        return get_matterport3d_pan_pipeline(config, split_name)
    else:
        raise NotImplementedError

def get_stanford2d3d_pan_pipeline(config, split_name='validation'):
    val_preprocess = ValPre()
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
    return Stanford2d3dPanDataset(setting=data_setting, split_name=split_name, preprocess=val_preprocess)

def get_structured3d_pan_pipeline(config, split_name='validation'):
    val_preprocess = ValPre()
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
    return Structured3dPanDataset(setting=data_setting, split_name=split_name, preprocess=val_preprocess)

def get_matterport3d_pan_pipeline(config, split_name='validation'):
    val_preprocess = ValPre()
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
    return Matterport3dPanDataset(setting=data_setting, split_name=split_name, preprocess=val_preprocess)

def get_ricoh3d_pan_pipeline(config, split_name='validation', mapping_name='Stanford2D3DS'):
    val_preprocess = ValPre()
    data_setting = {
        'dataset_path': config.dataset_path,
        'rgb': config.rgb,
        'ann': config.ann,
        'modality_x': config.modality_x,
        'ignore_index': config.ignore_index,
        'mask_black': ('dual' not in config.backbone and 'trio' not in config.backbone),
        'train_source': config.train_source,
        'eval_source': config.eval_source,
        'test_source': config.test_source,
    }
    return Ricoh3dPanDataset(setting=data_setting, split_name=split_name, mapping_name=mapping_name, preprocess=val_preprocess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs.sid')
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    config = import_module(args.config).config
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    if config.dataset_name == 'Stanford2D3DS':
        valid_pipeline = get_eval_pipeline('stanford2d3d_pan', config, split_name=args.split)
    elif config.dataset_name == 'Structured3D':
        valid_pipeline = get_eval_pipeline('structured3d_pan', config, split_name=args.split)
    elif config.dataset_name == 'Matterport3D':
        valid_pipeline = get_eval_pipeline('matterport3d_pan', config, split_name=args.split)
    else:
        raise NotImplementedError
 
    with torch.no_grad():
        segmentor = SegEvaluator(valid_pipeline, config.rgb, config.ann, config.num_classes, config.class_names, config.ignore_index,
                                 config.modality_x, config.eval_crop_size, config.eval_stride_rate,
                                 config.norm_mean, config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.val_func = network
        segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)