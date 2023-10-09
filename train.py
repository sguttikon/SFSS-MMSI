import os.path as osp
import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm

root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from dataloader.dataloader import get_train_pipeline
from models.builder import EncoderDecoder as segmodel
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from importlib import import_module
from eval import SegEvaluator, get_eval_pipeline
from utils.pyt_utils import parse_devices
from utils.visualize import print_iou
from fvcore.nn import flop_count_table, FlopCountAnalysis

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs.sid')
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = import_module(args.config).config

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    if config.dataset_name == 'Stanford2D3DS':
        train_loader, train_sampler = get_train_pipeline('stanford2d3d_pan', config)
        valid_pipeline = get_eval_pipeline('stanford2d3d_pan', config, split_name='validation')
    elif config.dataset_name == 'Structured3D':
        train_loader, train_sampler = get_train_pipeline('structured3d_pan', config)
        valid_pipeline = get_eval_pipeline('structured3d_pan', config, split_name='validation')
    elif config.dataset_name == 'Matterport3D':
        train_loader, train_sampler = get_train_pipeline('matterport3d_pan', config)
        valid_pipeline = get_eval_pipeline('matterport3d_pan', config, split_name='validation')
    else:
        raise NotImplementedError
    config.num_train_imgs = len(train_loader.dataset)
    config.num_eval_imgs = len(valid_pipeline)
    print('num_train_imgs: ', config.num_train_imgs)
    print('num_eval_imgs: ', config.num_eval_imgs)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.ignore_index)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    config.niters_per_epoch = config.num_train_imgs // config.batch_size  + 1
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # logger.info('================== model complexity =====================')
        # x = [torch.zeros(1, 3, 512, 512).to(device) for _ in range(len([config.rgb] + config.modality_x))]
        # logger.info(flop_count_table(FlopCountAnalysis(model, x))) 

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')

    evaluator = SegEvaluator(valid_pipeline, config.rgb, config.ann, config.num_classes, config.class_names,
                             config.ignore_index, config.modality_x, config.eval_crop_size, config.eval_stride_rate,
                             config.norm_mean, config.norm_std, model, config.eval_scale_array,
                             config.eval_flip, [None])    
    best_mIoU = 0.0
    best_epoch = 0
    valid_labels = np.arange(config.num_classes).tolist() + [config.ignore_index]
    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(enumerate(train_loader), total = config.niters_per_epoch, file=sys.stdout,
                    bar_format=bar_format)

        sum_loss = 0

        for idx, minibatch in pbar:
            engine.update_iteration(epoch, idx)

            imgs = minibatch[config.rgb].float()
            gts = minibatch[config.ann].long()
            modal_xs1 = minibatch[config.modality_x[0]].float()
            if len(config.modality_x) == 1:
                modal_xs2 = None
            elif len(config.modality_x) == 2:
                modal_xs2 = minibatch[config.modality_x[1]].float()
            else:
                raise NotImplementedError
            assert set(torch.unique(gts).tolist()).issubset(valid_labels), 'Unknown target label'

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs1 = modal_xs1.cuda(non_blocking=True)
            if modal_xs2 is not None:
                modal_xs2 = modal_xs2.cuda(non_blocking=True)

            aux_rate = 0.2
            loss = model(imgs, modal_xs1, modal_xs2, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))

            del loss
            pbar.set_description(print_str, refresh=False)
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):          
            evaluator.val_func = model
            iou_results = evaluator.single_process_evalutation()
            result_line = print_iou(iou_results['IoU'], 0,
                                    np.nanmean(iou_results['Acc']), iou_results['aAcc'],
                                    config.class_names[1:], show_no_back=False) # ignore background/unknown class
            miou = np.nanmean(iou_results['IoU'])
            if miou > best_mIoU:
                prev_best = osp.join(config.checkpoint_dir, 'epoch-{}-{}.pth'.format(best_epoch, best_mIoU))
                if os.path.isfile(prev_best): os.remove(prev_best)
                best_mIoU = miou
                best_epoch = epoch

                if engine.distributed and (engine.local_rank == 0):
                    engine.save_and_link_checkpoint(miou, config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
                elif not engine.distributed:
                    engine.save_and_link_checkpoint(miou, config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
            print(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU}")