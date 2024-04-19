# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import argparse

import json
from typing import Tuple
import numpy as np
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "500"

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from util.dataset import SignalDataset
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.callbacks import EarlyStop

import models_otis
from sklearn.linear_model import LogisticRegression, LinearRegression

from engine_pretrain import train_one_epoch, evaluate_online, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('OTiS pre-training', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='otis_baseDeep_dec128d2b_patchX', type=str, metavar='MODEL',
                        help='Name of model to train (default: otis_baseDeep_dec128d2b_patchX)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch compile')

    parser.add_argument('--input_channels', type=int, default=1, metavar='N',
                        help='input channels')
    parser.add_argument('--input_electrodes', type=int, default=12, metavar='N',
                        help='input electrodes')
    parser.add_argument('--time_steps', type=int, default=5000, metavar='N',
                        help='input length')
    parser.add_argument('--input_size', default=(1, 12, 5000), type=Tuple,
                        help='images input size')
                        
    parser.add_argument('--patch_height', type=int, default=1, metavar='N',
                        help='patch height')
    parser.add_argument('--patch_width', type=int, default=100, metavar='N',
                        help='patch width')
    parser.add_argument('--patch_size', default=(1, 100), type=Tuple,
                        help='patch size')

    parser.add_argument('--separate_dec_pos_embed_y', action='store_true', default=False,
                        help='Use separate position embeddings Y for the decoder')

    parser.add_argument('--norm_pix_loss', action='store_true', default=False,
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--masked_patch_loss', action='store_true', default=False,
                        help='Compute loss only on masked patches')
    parser.add_argument('--modality_weighted_loss', action='store_true', default=False,
                        help='Use weighted loss to consider imbalances between modalities')

    parser.add_argument('--ncc_weight', type=float, default=0.1,
                        help='Add normalized cross-correlation (ncc) as additional loss term')
    parser.add_argument('--cos_weight', type=float, default=0.0,
                        help='Add cos similarity as additional loss term')

    # Augmentation parameters
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--include_forecasting_mask', action='store_true', default=False,
                        help='Introduce also masking for forecasting (i.e. right-sided masking).')

    parser.add_argument('--crop_lower_bnd', default=0.5, type=float,
                        help='Lower boundary of the cropping ratio (default: 0.5)')
    parser.add_argument('--crop_upper_bnd', default=1.0, type=float,
                        help='Upper boundary of the cropping ratio (default: 1.0)')

    parser.add_argument('--jitter_sigma', default=0.2, type=float,
                        help='Jitter sigma N(0, sigma) (default: 0.2)')
    parser.add_argument('--rescaling_sigma', default=0.5, type=float,
                        help='Rescaling sigma N(0, sigma) (default: 0.5)')
    parser.add_argument('--ft_surr_phase_noise', default=0.075, type=float,
                        help='Phase noise magnitude (default: 0.075)')
    parser.add_argument('--freq_shift_delta', default=0.005, type=float,
                        help='Delta for the frequency shift (default: 0.005)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Callback parameters
    parser.add_argument('--patience', default=-1, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # Dataset parameters
    parser.add_argument('--data_path', default='_.pt', type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default='', type=str,
                        help='validation dataset path')
    
    parser.add_argument('--online_evaluation', action='store_true', default=False,
                        help='Perform online evaluation of a downstream task')
    parser.add_argument('--online_evaluation_task', default='classification', type=str,
                        help='Online downstream task (default: classification)')
    parser.add_argument('--online_num_classes', default=2, type=int,
                        help='Online classification task classes (default: 2)')
    
    parser.add_argument('--lower_bnd', type=int, default=0, metavar='N',
                        help='lower_bnd')
    parser.add_argument('--upper_bnd', type=int, default=0, metavar='N',
                        help='upper_bnd')

    parser.add_argument('--data_path_online', default='_.pt', type=str,
                        help='dataset path for the online evaluation')
    parser.add_argument('--labels_path_online', default='_.pt', type=str,
                        help='labels path for the online evaluation')
    parser.add_argument('--labels_mask_path_online', default='', type=str,
                        help='labels path (default: None)')
    
    parser.add_argument('--val_data_path_online', default='', type=str,
                        help='validation dataset path for the online evaluation')
    parser.add_argument('--val_labels_path_online', default='', type=str,
                        help='validation labels path for the online evaluation')
    parser.add_argument('--val_labels_mask_path_online', default='', type=str,
                        help='labels path (default: None)')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log (default: ./logs)')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', default='',
                        help='project where to wandb log')
    parser.add_argument('--wandb_id', default='', type=str,
                        help='id of the current run')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=24, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor)')

    return parser


def main(args):
    args.patch_size = (args.patch_height, args.patch_width)

    print(f"cuda devices: {torch.cuda.device_count()}")
    misc.init_distributed_mode(args)
    # args.distributed = False

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    print(f"rank: {misc.get_rank()}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # load data
    dataset_train = SignalDataset(data_path=args.data_path, train=True, args=args)
    dataset_val = SignalDataset(data_path=args.val_data_path, train=False, modality_offsets=dataset_train.offsets, args=args)

    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        print(f"num_tasks: {num_tasks}")
        global_rank = misc.get_rank()
        print(f"global_rank: {global_rank}")
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True # shuffle=True to reduce monitor bias
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        print("Sampler_val = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # tensorboard logging
    if False: #global_rank == 0 and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # wandb logging
    if args.wandb == True and misc.is_main_process():
        config = vars(args)
        if args.wandb_id:
            wandb.init(project=args.wandb_project, id=args.wandb_id, config=config, entity="oturgut")
        else:
            wandb.init(project=args.wandb_project, config=config, entity="oturgut")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        # shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset_train.collate_fn,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        # shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset_val.collate_fn,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # online evaluation
    if args.online_evaluation:
        dataset_online_train = SignalDataset(data_path=args.data_path_online, 
                                             labels_path=args.labels_path_online, 
                                             labels_mask_path=args.labels_mask_path_online, 
                                             downstream_task=args.online_evaluation_task, 
                                             train=True, 
                                             modality_offsets=dataset_train.offsets,
                                             args=args)
        dataset_online_val = SignalDataset(data_path=args.val_data_path_online, 
                                           labels_path=args.val_labels_path_online, 
                                           labels_mask_path=args.val_labels_mask_path_online, 
                                           downstream_task=args.online_evaluation_task, 
                                           train=False, 
                                           modality_offsets=dataset_train.offsets, 
                                           args=args)

        sampler_online_train = torch.utils.data.DistributedSampler(
            dataset_online_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_online_train = %s" % str(sampler_online_train))

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed online evaluation with an eval dataset not divisible '
                      'by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added '
                      'to achieve equal num of samples per-process.')
            sampler_online_val = torch.utils.data.DistributedSampler(
                dataset_online_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            ) 
        else:
            sampler_online_val = torch.utils.data.SequentialSampler(dataset_online_val)
        print("Sampler_online_val = %s" % str(sampler_online_val))

        data_loader_online_train = torch.utils.data.DataLoader(
            dataset_online_train, 
            sampler=sampler_online_train,
            shuffle=False,
            batch_size=256,
            num_workers=args.num_workers,
            collate_fn=dataset_online_train.collate_fn_ft,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        data_loader_online_val = torch.utils.data.DataLoader(
            dataset_online_val, 
            sampler=sampler_online_val,
            shuffle=False,
            batch_size=256,
            num_workers=args.num_workers,
            collate_fn=dataset_online_val.collate_fn_ft,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    # define the model
    model = models_otis.__dict__[args.model](
        modalities=dataset_train.modalities,
        modality_weights=dataset_train.modality_weights,
        input_channels=args.input_channels,
        time_steps=args.time_steps,
        patch_size=args.patch_size,
        separate_dec_pos_embed_y=args.separate_dec_pos_embed_y,
        norm_pix_loss=args.norm_pix_loss,
        masked_patch_loss=args.masked_patch_loss,
        modality_weighted_loss=args.modality_weighted_loss,
        ncc_weight=args.ncc_weight,
        include_forecasting_mask=args.include_forecasting_mask,
    )

    if args.compile:
        model = torch.compile(model)
    model.to(device, non_blocking=True)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_encoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "decoder" not in n)
    n_params_decoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "decoder" in n)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))
    print('Number of encoder params (M): %.2f' % (n_params_encoder / 1.e6))
    print('Number of decoder params (M): %.2f' % (n_params_decoder / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 32

    print("base lr: %.2e" % (args.lr * 32 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")

    eval_criterion = "ncc"
    
    best_stats = {'total_loss':np.inf, 'loss':np.inf, 'ncc':0.0, 'cos_sim':-1.0, 'mse':np.inf, 'mae':np.inf}
    best_eval_scores = {'count':0, 'nb_ckpts_max':5, 'eval_criterion':[best_stats[eval_criterion]]}
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        if True: #args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats, train_history = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler,
                                                     log_writer=log_writer, args=args)

        val_stats, val_history = evaluate(data_loader_val, model, device, epoch, 
                                          log_writer=log_writer, args=args)

        # online evaluation of the downstream task
        online_history = {}
        if args.online_evaluation and epoch % 5 == 0:
            if args.online_evaluation_task == "classification":
                online_estimator = LogisticRegression(class_weight='balanced', max_iter=2000)
            elif args.online_evaluation_task == "regression":
                online_estimator = LinearRegression()
            
            online_history = evaluate_online(estimator=online_estimator, model=model_without_ddp, device=device, 
                                             train_dataloader=data_loader_online_train, 
                                             val_dataloader=data_loader_online_val, args=args)
        
        if eval_criterion in ["total_loss", "loss", "mse", "mae"]:
            if early_stop.evaluate_decreasing_metric(val_metric=val_stats[eval_criterion]):
                break
            if args.output_dir and val_stats[eval_criterion] <= max(best_eval_scores['eval_criterion']):
                # save the best 5 (nb_ckpts_max) checkpoints, even if they appear after the best checkpoint wrt time
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'])
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(val_stats[eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion, 
                    mode="decreasing", modalities=dataset_train.modalities, modality_offsets=dataset_train.offsets)
        else:
            if early_stop.evaluate_increasing_metric(val_metric=val_stats[eval_criterion]):
                break
            if args.output_dir and val_stats[eval_criterion] >= min(best_eval_scores['eval_criterion']):
                # save the best 5 (nb_ckpts_max) checkpoints, even if they appear after the best checkpoint wrt time
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'], reverse=True)
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(val_stats[eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, evaluation_criterion=eval_criterion, 
                    mode="increasing", modalities=dataset_train.modalities, modality_offsets=dataset_train.offsets)
        
        best_stats['total_loss'] = min(best_stats['total_loss'], val_stats['total_loss'])
        best_stats['loss'] = min(best_stats['loss'], val_stats['loss'])
        best_stats['ncc'] = max(best_stats['ncc'], val_stats['ncc'])
        best_stats['cos_sim'] = max(best_stats['cos_sim'], val_stats['cos_sim'])
        best_stats['mse'] = min(best_stats['mse'], val_stats['mse'])
        best_stats['mae'] = min(best_stats['mae'], val_stats['mae'])

        print(f"Total Loss / Loss / Normalized Cross-Correlation (NCC) / Cosine Similarity / Mean Squared Error (MSE) / Mean Absolute Error (MAE)",
              f"of the network on {len(dataset_val)} val images: {val_stats['total_loss']:.4f} / {val_stats['loss']:.4f} / ", 
              f"{val_stats['ncc']:.2f} / {val_stats['cos_sim']:.2f} / {val_stats['mse']:.2f} / {val_stats['mae']:.2f}")

        print(f"Min Total Loss / Min Loss / Max NCC / Max Cosine Similarity / Min MSE / Min MAE: ",
              f"{best_stats['total_loss']:.4f} / {best_stats['loss']:.4f} / {best_stats['ncc']:.2f} / ", 
              f"{best_stats['cos_sim']:.2f} / {best_stats['mse']:.2f} / {best_stats['mae']:.2f}\n")
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch, 
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        if args.wandb and misc.is_main_process():
            wandb.log(train_history | val_history | online_history | {"Time per epoch [sec]": total_time})

    if args.wandb and misc.is_main_process():
        wandb.log({f'Best Statistics/{k}': v for k, v in best_stats.items()})
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)