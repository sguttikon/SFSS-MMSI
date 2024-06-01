import os
import json
from pickletools import uint8
import cv2
import torch
import numpy as np
import PIL.Image

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._dataset_name = setting['dataset_name']
        self._dataset_path = setting['dataset_path']
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._gt_path = setting['gt_root']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

        with open('semantic_labels.json') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open('name2label.json') as f:
            name2id = json.load(f)
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            if self._dataset_name == 'Stanford2D3DS':
                item_area, item_name = self._construct_new_file_names(self._file_length)[index].strip().split(' ')
            else:
                item_area, item_name = '', self._construct_new_file_names(self._file_length)[index]
        else:
            if self._dataset_name == 'Stanford2D3DS':
                item_area, item_name = self._file_names[index].strip().split(' ')
            else:
                item_area, item_name = '', self._file_names[index]
        rgb_path = os.path.join(self._dataset_path, item_area, 'pano', self._rgb_path, item_name + '_' + self._rgb_path + self._rgb_format)
        x_path = os.path.join(self._dataset_path, item_area, 'pano', self._x_path, item_name + '_' + self._x_path + self._x_format)
        gt_path = os.path.join(self._dataset_path, item_area, 'pano', self._gt_path, item_name + '_' + self._gt_path +  self._gt_format)

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        gt = self._open_image(gt_path, cv2.IMREAD_COLOR)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        unk = (gt[..., 0] != 0)
        gt = self.id2label[gt[..., 1] * 256 + gt[..., 2]]
        gt[unk] = 0
        if self._transform_gt:
            gt = self._gt_transform(gt) 

        if self._x_single_channel:
            x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE)
            x = cv2.merge([x, x, x])
        else:
            x =  self._open_image(x_path, cv2.IMREAD_COLOR)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        if self._split_name == 'train':
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()

        output_dict = dict(data=rgb, label=gt, modal_x=x, fn=str(item_name), n=len(self._file_names))

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)                          
        new_file_names = self._file_names * (length // files_len)   

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

_image_specs = {
    "u8": ("numpy", "uint8", "raw"),
    "i": ("numpy", "float", "i"),
    "rgb": ("numpy", "float", "rgb"),
}

def image_decoder(img_path, image_spec):
    a_type, e_type, mode = _image_specs[image_spec]
    image = PIL.Image.open(img_path)

    if mode != "raw":
        image = image.convert(mode.upper())

    image = np.asarray(image, dtype=np.uint16 if image.mode == "I" else np.uint8)

    if a_type == "numpy":
        if e_type == "float":
            image = image.astype("f") / np.iinfo(image.dtype).max
        return image
    return None

class Stanford2d3dPanDataset(data.Dataset):

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(Stanford2d3dPanDataset, self).__init__()
        assert split_name in ['train', 'validation']
        self.dataset_path = setting['dataset_path']
        self.rgb = setting['rgb']
        self.ann = setting['ann']
        self.modality_x = setting['modality_x']
        self.mask_black = setting['mask_black']
        self.ignore_index = setting['ignore_index']
        self.source = setting['eval_source'] if split_name == "validation" else setting['train_source']
        self.file_names = self._get_file_names(split_name)
        self.file_length = file_length
        self.preprocess = preprocess

        with open(os.path.join(self.dataset_path, 'assets/semantic_labels.json'), 'r', encoding='utf8') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open(os.path.join(self.dataset_path, 'assets/name2label.json'), 'r', encoding='utf8') as f:
            name2id = json.load(f)
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)

    def __len__(self):
        if self.file_length is not None:
            return self.file_length
        return len(self.file_names)

    def __getitem__(self, index):
        out = {}
        if self.file_length is not None:
            raise NotImplementedError
        else:
            item_area, item_name = self.file_names[index].strip().split(' ')
        out['sample_token'] = f'{item_area}/{item_name}'

        rgb_path = os.path.join(self.dataset_path, item_area, 'pano/rgb', item_name + '_rgb.png')
        data = image_decoder(rgb_path, 'rgb')
        out['camera-rgb-1K'] = np.array(data * 255.0, np.uint8) # (H, W, 3)

        gt_path = os.path.join(self.dataset_path, item_area, 'pano/semantic', item_name + '_semantic.png')
        data = image_decoder(gt_path, 'rgb')
        semantic = np.array(data * 255.0, np.int32)
        unk = (semantic[..., 0] != 0)
        semantic = self.id2label[semantic[..., 1] * 256 + semantic[..., 2]]
        semantic[unk] = 0
        out['camera-semantic-1K'] = semantic - 1  # (H, W) # transform unknown id: 0 -> 255

        for component_name in self.modality_x:
            if component_name == 'camera-hha-1K':
                x_path = os.path.join(self.dataset_path, item_area, 'pano/hha', item_name + '_hha.png')
                data = image_decoder(x_path, 'rgb')
                out[component_name] = np.array(data * 255.0, np.uint8) # (H, W, 3)
            elif component_name == 'camera-normals-1K':
                x_path = os.path.join(self.dataset_path, item_area, 'pano/normal', item_name + '_normals.png')
                data = image_decoder(x_path, 'rgb')
                out[component_name] = np.array(data * 255.0, np.uint8) # (H, W, 3)
            elif component_name == 'camera-depth-1K':
                x_path = os.path.join(self.dataset_path, item_area, 'pano/depth', item_name + '_depth.png')
                data = image_decoder(x_path, 'i')
                x = np.array(data * 255.0, np.uint8)
                # ignore max depth (65535 -> 0)
                x = np.where(x == 255, 0, x)
                if x.ndim == 2:
                    # single channel -> 3 channels
                    out[component_name] = cv2.merge([x, x, x]) # (H, W, 3)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        if self.mask_black:
            out['camera-semantic-1K'][out['camera-rgb-1K'].sum(-1) == 0] = self.ignore_index  # mask as unknown id: 255
        out['camera-semantic-1K'][out['camera-semantic-1K'] == 255] = self.ignore_index  # mask as unknown id: 255

        if self.preprocess is not None:
            if len(self.modality_x) == 1:
                rgb, gt, x1 = out['camera-rgb-1K'], out['camera-semantic-1K'], out[self.modality_x[0]]
                rgb, gt, x1, _ = self.preprocess(rgb, gt, x1, x1)
                out['camera-rgb-1K'], out['camera-semantic-1K'], out[self.modality_x[0]] = rgb, gt, x1
            elif len(self.modality_x) == 2:
                rgb, gt, x1, x2 = out['camera-rgb-1K'], out['camera-semantic-1K'], out[self.modality_x[0]], out[self.modality_x[1]]
                rgb, gt, x1, x2 = self.preprocess(rgb, gt, x1, x2)
                out['camera-rgb-1K'], out['camera-semantic-1K'], out[self.modality_x[0]], out[self.modality_x[1]] = rgb, gt, x1, x2
            else:
                raise NotImplementedError   
        return out

    def _get_file_names(self, split_name):
        file_names = []
        with open(self.source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

class Structured3dPanDataset(data.Dataset):

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(Structured3dPanDataset, self).__init__()
        assert split_name in ['train', 'validation', 'test']
        self.dataset_path = setting['dataset_path']
        self.area = setting['area']
        self.lighting = setting['lighting']
        self.rgb = setting['rgb']
        self.ann = setting['ann']
        self.modality_x = setting['modality_x']
        self.mask_black = setting['mask_black']
        self.ignore_index = setting['ignore_index']
        self.source = setting['test_source'] if split_name == "test" else (setting['eval_source'] if split_name == "validation" else setting['train_source'])
        self.file_names = self._get_file_names(split_name)
        self.file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self.file_length is not None:
            return self.file_length
        return len(self.file_names)

    def __getitem__(self, index):
        out = {}
        if self.file_length is not None:
            raise NotImplementedError
        else:
            item_area, item_name = self.file_names[index].strip().split(' ')
        out['sample_token'] = f'{item_area}/{item_name}'

        rgb_path = os.path.join(self.dataset_path, item_area, '2D_rendering', item_name, 'panorama', self.area, f'rgb_{self.lighting}.png')
        data = image_decoder(rgb_path, 'rgb')
        out[f'{self.area}-rgb-{self.lighting}-1K'] = np.array(data * 255.0, np.uint8) # (H, W, 3)

        gt_path = os.path.join(self.dataset_path, item_area, '2D_rendering', item_name, 'panorama', self.area, 'semantic.png')
        data = image_decoder(gt_path, 'u8')
        out[f'{self.area}-semantic-1K'] = data - 1  # (H, W) # transform unknown id: 0 -> 255

        for component_name in self.modality_x:
            if component_name == f'{self.area}-normal-1K':
                x_path = os.path.join(self.dataset_path, item_area, '2D_rendering', item_name, 'panorama', self.area, 'normal.png')
                data = image_decoder(x_path, 'rgb')
                out[component_name] = np.array(data * 255.0, np.uint8) # (H, W, 3)
            elif component_name == f'{self.area}-depth-1K':
                x_path = os.path.join(self.dataset_path, item_area, '2D_rendering', item_name, 'panorama', self.area, 'depth.png')
                data = image_decoder(x_path, 'i')
                x = np.array(data * 255.0, np.uint8)
                # ignore max depth (65535 -> 0)
                x = np.where(x == 255, 0, x)
                if x.ndim == 2:
                    # single channel -> 3 channels
                    out[component_name] = cv2.merge([x, x, x]) # (H, W, 3)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
        if self.mask_black:
            out[f'{self.area}-semantic-1K'][out[f'{self.area}-rgb-{self.lighting}-1K'].sum(-1) == 0] = self.ignore_index  # mask as unknown id: 255
        out[f'{self.area}-semantic-1K'][out[f'{self.area}-semantic-1K'] == 255] = self.ignore_index  # mask as unknown id: 255

        if self.preprocess is not None:
            if len(self.modality_x) == 1:
                rgb, gt, x1 = out[f'{self.area}-rgb-{self.lighting}-1K'], out[f'{self.area}-semantic-1K'], out[self.modality_x[0]]
                rgb, gt, x1, _ = self.preprocess(rgb, gt, x1, x1)
                out[f'{self.area}-rgb-{self.lighting}-1K'], out[f'{self.area}-semantic-1K'], out[self.modality_x[0]] = rgb, gt, x1
            elif len(self.modality_x) == 2:
                rgb, gt, x1, x2 = out[f'{self.area}-rgb-{self.lighting}-1K'], out[f'{self.area}-semantic-1K'], out[self.modality_x[0]], out[self.modality_x[1]]
                rgb, gt, x1, x2 = self.preprocess(rgb, gt, x1, x2)
                out[f'{self.area}-rgb-{self.lighting}-1K'], out[f'{self.area}-semantic-1K'], out[self.modality_x[0]], out[self.modality_x[1]] = rgb, gt, x1, x2
            else:
                raise NotImplementedError
        return out

    def _get_file_names(self, split_name):
        file_names = []
        with open(self.source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

class Matterport3dPanDataset(data.Dataset):

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(Matterport3dPanDataset, self).__init__()
        assert split_name in ['train', 'validation', 'test']
        self.dataset_path = setting['dataset_path']
        self.rgb = setting['rgb']
        self.ann = setting['ann']
        self.modality_x = setting['modality_x']
        self.mask_black = setting['mask_black']
        self.ignore_index = setting['ignore_index']
        self.source = setting['test_source'] if split_name == "test" else (setting['eval_source'] if split_name == "validation" else setting['train_source'])
        self.file_names = self._get_file_names(split_name)
        self.file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self.file_length is not None:
            return self.file_length
        return len(self.file_names)

    def __getitem__(self, index):
        out = {}
        if self.file_length is not None:
            raise NotImplementedError
        else:
            item_area, item_name = self.file_names[index].strip().split(' ')
        out['sample_token'] = f'{item_area}/{item_name}'

        rgb_path = os.path.join(self.dataset_path, item_area, 'panorama', 'undistorted_color_images', f'{item_name}.png')
        data = image_decoder(rgb_path, 'rgb')
        out['rgb-1K'] = np.array(data * 255.0, np.uint8) # (H, W, 3)

        gt_path = os.path.join(self.dataset_path, item_area, 'panorama', 'segmentation_maps_classes', f'{item_name}.png')
        data = image_decoder(gt_path, 'u8')
        out['semantic-1K'] = data - 1  # (H, W) # transform unknown id: 0 -> 255

        for component_name in self.modality_x:
            if component_name == 'normal-1K':
                x_path = os.path.join(self.dataset_path, item_area, 'panorama', 'undistorted_normal_images_processed', f'{item_name}.png')
                data = image_decoder(x_path, 'rgb')
                out[component_name] = np.array(data * 255.0, np.uint8) # (H, W, 3)
            elif component_name == 'depth-1K':
                x_path = os.path.join(self.dataset_path, item_area, 'panorama', 'undistorted_depth_images', f'{item_name}.png')
                data = image_decoder(x_path, 'i')
                x = np.array(data * 255.0, np.uint8)
                # ignore max depth (65535 -> 0)
                x = np.where(x == 255, 0, x)
                if x.ndim == 2:
                    # single channel -> 3 channels
                    out[component_name] = cv2.merge([x, x, x]) # (H, W, 3)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
        if self.mask_black:
            out['semantic-1K'][out['rgb-1K'].sum(-1) == 0] = self.ignore_index  # mask as unknown id: 255
        out['semantic-1K'][out['semantic-1K'] == 255] = self.ignore_index  # mask as unknown id: 255

        if self.preprocess is not None:
            if len(self.modality_x) == 1:
                rgb, gt, x1 = out['rgb-1K'], out['semantic-1K'], out[self.modality_x[0]]
                rgb, gt, x1, _ = self.preprocess(rgb, gt, x1, x1)
                out['rgb-1K'], out['semantic-1K'], out[self.modality_x[0]] = rgb, gt, x1
            elif len(self.modality_x) == 2:
                rgb, gt, x1, x2 = out['rgb-1K'], out['semantic-1K'], out[self.modality_x[0]], out[self.modality_x[1]]
                rgb, gt, x1, x2 = self.preprocess(rgb, gt, x1, x2)
                out['rgb-1K'], out['semantic-1K'], out[self.modality_x[0]], out[self.modality_x[1]] = rgb, gt, x1, x2
            else:
                raise NotImplementedError
        return out

    def _get_file_names(self, split_name):
        file_names = []
        with open(self.source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

class Ricoh3dPanDataset(data.Dataset):

    def __init__(self, setting, split_name, mapping_name, preprocess=None, file_length=None):
        super(Ricoh3dPanDataset, self).__init__()
        assert split_name in ['train', 'validation', 'test']
        self.dataset_path = setting['dataset_path']
        self.rgb = setting['rgb']
        self.ann = setting['ann']
        self.modality_x = setting['modality_x']
        self.mask_black = setting['mask_black']
        self.ignore_index = setting['ignore_index']
        self.source = setting['test_source'] if split_name == "test" else (setting['eval_source'] if split_name == "validation" else setting['train_source'])
        self.file_names = self._get_file_names(split_name)
        self.file_length = file_length
        self.preprocess = preprocess

        if mapping_name == 'Stanford2D3DS':
            with open(os.path.join(self.dataset_path, 'assets/2d3dsmapping.json'), 'r', encoding='utf8') as f:
                self.mapping_json = json.load(f)
        elif mapping_name == 'Structured3D':
            with open(os.path.join(self.dataset_path, 'assets/structured3dmapping.json'), 'r', encoding='utf8') as f:
                self.mapping_json = json.load(f)
        else:
            raise NotImplementedError

    def __len__(self):
        if self.file_length is not None:
            return self.file_length
        return len(self.file_names)

    def __getitem__(self, index):
        out = {}
        if self.file_length is not None:
            raise NotImplementedError
        else:
            item_area, item_name = self.file_names[index].strip().split(' ')
        out['sample_token'] = f'{item_area}/{item_name}'

        rgb_path = os.path.join(self.dataset_path, item_area, 'pano/rgb', item_name + '_rgb.png')
        data = image_decoder(rgb_path, 'rgb')
        out['camera-rgb-1K'] = np.array(data * 255.0, np.uint8) # (H, W, 3)

        gt_path = os.path.join(self.dataset_path, item_area, 'pano/semantic', item_name + '_semantic.npy')
        with open(gt_path, 'rb') as f:
            data = np.load(gt_path)
        semantic = np.stack([np.array([self.mapping_json[str(l)] for l in r], dtype=np.uint8) for r in data])
        out['camera-semantic-1K'] = semantic - 1  # (H, W) # transform unknown id: 0 -> 255

        for component_name in self.modality_x:
            if component_name == 'camera-hha-1K':
                x_path = os.path.join(self.dataset_path, item_area, 'pano/hha', item_name + '_hha.png')
                data = image_decoder(x_path, 'rgb')
                out[component_name] = np.array(data * 255.0, np.uint8) # (H, W, 3)
            elif component_name == 'camera-normal-1K':
                x_path = os.path.join(self.dataset_path, item_area, 'pano/normal', item_name + '_normal.png')
                data = image_decoder(x_path, 'rgb')
                out[component_name] = np.array(data * 255.0, np.uint8) # (H, W, 3)
            elif component_name == 'camera-depth-1K':
                x_path = os.path.join(self.dataset_path, item_area, 'pano/depth', item_name + '_depth.png')
                data = image_decoder(x_path, 'i')
                x = np.array(data * 255.0, np.uint8)
                # ignore max depth (65535 -> 0)
                x = np.where(x == 255, 0, x)
                if x.ndim == 2:
                    # single channel -> 3 channels
                    out[component_name] = cv2.merge([x, x, x]) # (H, W, 3)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        if self.mask_black:
            out['camera-semantic-1K'][out['camera-rgb-1K'].sum(-1) == 0] = self.ignore_index  # mask as unknown id: 255
        out['camera-semantic-1K'][out['camera-semantic-1K'] == 255] = self.ignore_index  # mask as unknown id: 255

        if self.preprocess is not None:
            if len(self.modality_x) == 1:
                rgb, gt, x1 = out['camera-rgb-1K'], out['camera-semantic-1K'], out[self.modality_x[0]]
                rgb, gt, x1, _ = self.preprocess(rgb, gt, x1, x1)
                out['camera-rgb-1K'], out['camera-semantic-1K'], out[self.modality_x[0]] = rgb, gt, x1
            elif len(self.modality_x) == 2:
                rgb, gt, x1, x2 = out['camera-rgb-1K'], out['camera-semantic-1K'], out[self.modality_x[0]], out[self.modality_x[1]]
                rgb, gt, x1, x2 = self.preprocess(rgb, gt, x1, x2)
                out['camera-rgb-1K'], out['camera-semantic-1K'], out[self.modality_x[0]], out[self.modality_x[1]] = rgb, gt, x1, x2
            else:
                raise NotImplementedError   
        return out

    def _get_file_names(self, split_name):
        file_names = []
        with open(self.source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names