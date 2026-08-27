# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae?tab=readme-ov-file
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import argparse

import json
import sys
from typing import Tuple
import numpy as np
# Allow loading object-dtype .npy files pickled under numpy 2.x (which references
# the renamed ``numpy._core`` module path) when running under numpy 1.x. The
# legacy domain metadata in our mmap data shards (e.g. ``domain_names.npy``) was
# saved with numpy 2.x; this shim lets a numpy 1.x env read them.
if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    if hasattr(np.core, "_multiarray_umath"):
        sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "500"

from util.dataset import TimeSeriesDataset, TimeSeriesDatasetMmap
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed_x
from util.callbacks import EarlyStop
from util.token_l2_norms import (track_token_l2_norms, plot_token_l2_norms,
                                 plot_last_layer_norm_evolution)

import models_otis
from sklearn.linear_model import LogisticRegression, LinearRegression

from engine_pretrain import (train_one_epoch, evaluate_online, evaluate,
                             evaluate_online_kshot, evaluate_online_forecast)


def _ts_dataset_cls(path: str):
    """Auto-pick dataset class: mmap directory vs ``.pt`` file."""
    return TimeSeriesDatasetMmap if os.path.isdir(path) else TimeSeriesDataset


def _token_norm_collate(batch):
    """Strip pos_embed_y + domain from the standard 4-tuple collate, since
    :func:`track_token_l2_norms` iterates ``(samples, attn_mask)`` pairs."""
    data, attn_mask, _pos_embed_y, _domain = TimeSeriesDataset.collate_fn(batch)
    return data, attn_mask


def get_args_parser():
    parser = argparse.ArgumentParser('OTIS pre-training', add_help=False)
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

    parser.add_argument('--univariate', action='store_true', default=False,
                        help='Univariate time series analysis (i.e. treat each variate independently)')
    
    parser.add_argument('--domain_agnostic', action='store_true', default=False,
                        help='Share position embedding Y across all domains')

    parser.add_argument('--amp_dtype', default='fp16', type=str, choices=['fp16', 'bf16'],
                        help='CUDA AMP autocast dtype for the training forward pass (default: fp16). '
                             'GradScaler is auto-disabled under bf16.')

    parser.add_argument('--use_swiglu', action='store_true', default=False,
                        help='Use SwiGLU MLP with SiLU activation in encoder and decoder blocks (default: Mlp + GELU)')
    parser.add_argument('--layer_decay', type=float, default=0.95,
                        help='Layer-wise LR decay factor (default 0.95; set to >=1.0 to disable)')

    parser.add_argument('--input_channels', type=int, default=1, metavar='N',
                        help='input channels')
    parser.add_argument('--input_variates', type=int, default=12, metavar='N',
                        help='input variates')
    parser.add_argument('--time_steps', type=int, default=5000, metavar='N',
                        help='input length')
    parser.add_argument('--input_size', default=(1, 12, 5000), type=Tuple,
                        help='samples input size')
                        
    parser.add_argument('--patch_height', type=int, default=1, metavar='N',
                        help='patch height')
    parser.add_argument('--patch_width', type=int, default=100, metavar='N',
                        help='patch width')
    parser.add_argument('--patch_size', default=(1, 100), type=Tuple,
                        help='patch size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate for encoder (default: 0.1)')
    parser.add_argument('--drop_path_decoder', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate for decoder (default: 0.0)')

    parser.add_argument('--separate_dec_pos_embed_y', action='store_true', default=False,
                        help='Use separate position embeddings Y for the decoder')

    parser.add_argument('--norm_pix_loss', action='store_true', default=False,
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--masked_patch_loss', action='store_true', default=False,
                        help='Compute loss only on masked patches')
    parser.add_argument('--domain_weighted_loss', action='store_true', default=False,
                        help='Use weighted loss to consider imbalances between domains')

    parser.add_argument('--ncc_weight', type=float, default=0.1,
                        help='Add normalized cross-correlation (ncc) as additional loss term')
    parser.add_argument('--cos_weight', type=float, default=0.0,
                        help='Add cos similarity as additional loss term')

    # Augmentation parameters
    parser.add_argument('--probabilistic_masking', action='store_true', default=False,
                        help='Randomly vary the masking ratio during pretraining.')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--include_forecasting', action='store_true', default=False,
                        help='Include forecasting during pretraining (i.e. right-sided masking).')
    parser.add_argument('--forecasting_probability', default=0.33, type=float,
                        help='Probability for forecasting (i.e. right-sided masking).')
    parser.add_argument('--forecasting_mask_ratio', default=0.5, type=float,
                        help='Masking ratio for forecasting (percentage of removed patches).')

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
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 32')
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (should be 0.1x of peak lr)')

    parser.add_argument('--lr_schedule', default='wsd', type=str, choices=['wsd', 'cosine'],
                        help='learning rate schedule: warmup-stable-decay (default) or cosine')
    parser.add_argument('--warmup_fraction', type=float, default=0.1,
                        help='fraction of total optimizer steps used for linear warmup (default: 0.1)')
    parser.add_argument('--decay_fraction', type=float, default=0.1,
                        help='WSD only: fraction of total optimizer steps used for the cosine decay phase (default: 0.1)')

    # Callback parameters
    parser.add_argument('--patience', default=-1, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # * Finetuning params
    parser.add_argument('--pretrained_encoder', default='',
                        help='load encoder from checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='make encoder (i.e. the feature extractor) non-trainable, i.e., only train the decoder')
    parser.add_argument('--ignore_pos_embed_y', action='store_true', default=False,
                        help='ignore pretrained position embeddings Y (spatial axis)')
    
    # Dataset parameters
    eval_criterions = ['epoch', 'total_loss', 'loss', 'ncc', 'cos_sim', 'mse', 'mae', 'patch_patch_sim', 'rankme', 'kshot_cls_acc']
    parser.add_argument('--eval_criterion', default='ncc', type=str, choices=eval_criterions,
                        help='pretraining evaluation metric (default: ncc)')
    
    parser.add_argument('--data_path', default='_.pt', type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default='', type=str,
                        help='validation dataset path')

    # Fig.1 diagnostics — prototype k-shot UEA, synthetic forecasting, token L2 norms.
    parser.add_argument('--online_kshot', action='store_true', default=False,
                        help='Run prototypical k-shot classification over UEA datasets every 2 epochs.')
    parser.add_argument('--online_kshot_data_base', default='', type=str,
                        help='Base directory holding the UEA k-shot datasets.')
    parser.add_argument('--online_forecast', action='store_true', default=False,
                        help='Run a synthetic right-sided-masking forecasting eval every 2 epochs.')
    parser.add_argument('--online_forecast_data_path', default='', type=str,
                        help='Path to the synthetic forecasting dataset (.pt).')
    parser.add_argument('--online_forecast_time_steps', default=2400, type=int,
                        help='Time steps used for synthetic forecasting eval.')
    parser.add_argument('--online_forecast_mask_ratio', default=0.25, type=float,
                        help='Right-sided mask ratio for synthetic forecasting eval.')

    parser.add_argument('--track_token_norms', action='store_true', default=False,
                        help='Log per-layer patch-token L2-norm density diagrams every N epochs.')
    parser.add_argument('--track_token_norms_samples', default=4096, type=int,
                        help='Number of samples to feed through the token-norm diagnostic.')
    parser.add_argument('--track_token_norms_freq', default=2, type=int,
                        help='Run the token-norm diagnostic every N epochs.')

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
    parser.add_argument('--wandb_entity', default='', type=str,
                        help='entity of the current run')
    parser.add_argument('--wandb_project', default='', type=str,
                        help='project where to wandb log')
    parser.add_argument('--wandb_id', default='', type=str,
                        help='id of the current run')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--save_embeddings', action='store_true', default=False,
                        help='save encoder embeddings (i.e. of visible tokens)')

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
    # domain_offsets are initialized in dataset_train
    TrainCls = _ts_dataset_cls(args.data_path)
    dataset_train = TrainCls(data_path=args.data_path,
                             domain_agnostic=args.domain_agnostic,
                             univariate=args.univariate,
                             train=True,
                             args=args)
    print("Training set size: ", len(dataset_train))

    if args.val_data_path:
        ValCls = _ts_dataset_cls(args.val_data_path)
        dataset_val = ValCls(data_path=args.val_data_path,
                             domain_offsets=dataset_train.offsets,
                             univariate=args.univariate,
                             train=False,
                             args=args)
        print("Validation set size: ", len(dataset_val))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        print(f"num_tasks: {num_tasks}")
        global_rank = misc.get_rank()
        print(f"global_rank: {global_rank}")
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        # print("Sampler_train = %s" % str(sampler_train))

        if args.val_data_path:
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)  # shuffle=True to reduce monitor bias
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # print("Sampler_val = %s" % str(sampler_train))
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
            wandb.init(project=args.wandb_project, id=args.wandb_id, config=config, entity=args.wandb_entity)
        else:
            wandb.init(project=args.wandb_project, config=config, entity=args.wandb_entity)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        # shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset_train.collate_fn,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True,
    )

    if args.val_data_path:
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

    # Token L2-norm diagnostic loader — re-uses the training dataset but
    # emits only ``(samples, attn_mask)`` for :func:`track_token_l2_norms`.
    if args.track_token_norms:
        data_loader_token_norms = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_token_norm_collate,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    # History of last-block (pre-final-norm) patch-token L2 norms, one entry
    # per diagnostic call. Used at end-of-training to render a Fig 4(b)-style
    # multi-curve density plot showing the gradual emergence of the high-norm
    # outlier tail across pre-training.
    last_layer_norm_history_enc = []
    last_layer_norm_history_dec = []
    last_layer_norm_enc_dim = None
    last_layer_norm_dec_dim = None

    # Final-epoch full-stack snapshot, persisted to token_norms_final.pt so the
    # multi-layer "Token L2 Norms" panel can be regenerated offline.
    final_token_norms_enc = None
    final_token_norms_dec = None
    final_token_norms_epoch = None

    # online evaluation
    if args.online_evaluation:
        dataset_online_train = TimeSeriesDataset(data_path=args.data_path_online, 
                                                 labels_path=args.labels_path_online, 
                                                 labels_mask_path=args.labels_mask_path_online, 
                                                 downstream_task=args.online_evaluation_task, 
                                                 univariate=args.univariate,
                                                 train=True, 
                                                 args=args)
        dataset_online_val = TimeSeriesDataset(data_path=args.val_data_path_online, 
                                               labels_path=args.val_labels_path_online, 
                                               labels_mask_path=args.val_labels_mask_path_online, 
                                               downstream_task=args.online_evaluation_task, 
                                               domain_offsets=dataset_online_train.offsets, 
                                               univariate=args.univariate,
                                               train=False, 
                                               N_val=5,
                                               args=args)

        print("Online training set size: ", len(dataset_online_train))
        print("Online validation set size: ", len(dataset_online_val))

        sampler_online_train = torch.utils.data.DistributedSampler(
            dataset_online_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        # print("Sampler_online_train = %s" % str(sampler_online_train))

        if args.dist_eval:
            if len(dataset_online_val) % num_tasks != 0:
                print('Warning: Enabling distributed online evaluation with an eval dataset not divisible '
                      'by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added '
                      'to achieve equal num of samples per-process.')
            sampler_online_val = torch.utils.data.DistributedSampler(
                dataset_online_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            ) 
        else:
            sampler_online_val = torch.utils.data.SequentialSampler(dataset_online_val)
        # print("Sampler_online_val = %s" % str(sampler_online_val))

        data_loader_online_train = torch.utils.data.DataLoader(
            dataset_online_train, 
            sampler=sampler_online_train,
            shuffle=False,
            batch_size=128,
            num_workers=args.num_workers,
            collate_fn=dataset_online_train.collate_fn_ft,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        data_loader_online_val = torch.utils.data.DataLoader(
            dataset_online_val, 
            sampler=sampler_online_val,
            shuffle=False,
            batch_size=128,
            num_workers=args.num_workers,
            collate_fn=dataset_online_val.collate_fn_ft,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    # define the model
    model = models_otis.__dict__[args.model](
        domains=dataset_train.domains,
        domain_weights=dataset_train.domain_weights,
        domain_agnostic=args.domain_agnostic,
        input_channels=args.input_channels,
        time_steps=args.time_steps,
        patch_size=args.patch_size,
        separate_dec_pos_embed_y=args.separate_dec_pos_embed_y,
        norm_pix_loss=args.norm_pix_loss,
        masked_patch_loss=args.masked_patch_loss,
        domain_weighted_loss=args.domain_weighted_loss,
        contrastive_loss=(args.cos_weight > 0.0),
        probabilistic_masking=args.probabilistic_masking,
        include_forecasting=args.include_forecasting,
        forecasting_probability=args.forecasting_probability,
        forecasting_mask_ratio=args.forecasting_mask_ratio,
        drop_path=args.drop_path,
        drop_path_decoder=args.drop_path_decoder,
        use_swiglu=args.use_swiglu,
    )

    new_patch_size = False
    if args.pretrained_encoder:
        checkpoint = torch.load(args.pretrained_encoder, map_location='cpu', weights_only=False)

        print("Load pretrained encoder from: %s" % args.pretrained_encoder)
        checkpoint_model = checkpoint['model']

        # check if new and old patch_size match
        nb_channels_ckpt = checkpoint_model['patch_embed.proj.weight'].shape[-3]
        nb_channels_model = args.input_size[0]

        checkpoint_patch_size = checkpoint_model['patch_embed.proj.weight'].shape[-2:]
        patch_height_ckpt, patch_width_ckpt = checkpoint_patch_size[0], checkpoint_patch_size[1]
        patch_height_model, patch_width_model = args.patch_size[0], args.patch_size[1]

        if nb_channels_ckpt != nb_channels_model or patch_height_ckpt != patch_height_model or patch_width_ckpt != patch_width_model:
            new_patch_size = True
            # initialize new patch_embed module
            for key in ["patch_embed.proj.weight", "patch_embed.proj.bias", 
                        "patch_embed.norm.weight", "patch_embed.norm.bias"]:
                if key in checkpoint_model:
                    print(f"Removing key {key} from pretrained checkpoint")
                    del checkpoint_model[key]
            print("Initializing new patch_embed")

            # initialize new decoder_pred module
            for key in ["decoder_pred.weight", "decoder_pred.bias"]:
                if key in checkpoint_model:
                    print(f"Removing key {key} from pretrained checkpoint")
                    del checkpoint_model[key]
            print("Initializing new decoder_pred")

        # load pos_embed_x
        interpolate_pos_embed_x(model, checkpoint_model)

        key = "pos_embed_x"
        if key in checkpoint_model:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]

        # load pos_embed_y together with domain_offsets
        if not args.ignore_pos_embed_y:
            print("Loading pos_embed_y from checkpoint")
            model.pos_embed_y = torch.nn.Embedding.from_pretrained(checkpoint_model["pos_embed_y.weight"])

            # load domain_offsets
            dataset_train.set_domain_offsets(checkpoint["domain_offsets"])
            dataset_val.set_domain_offsets(checkpoint["domain_offsets"])

            if args.online_evaluation:
                dataset_online_train.set_domain_offsets(checkpoint["domain_offsets"])
                dataset_online_val.set_domain_offsets(checkpoint["domain_offsets"])
        else:
            print("Initializing new pos_embed_y")

        key = "pos_embed_y.weight"
        if key in checkpoint_model:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]

        # initialize new decoder
        print("Initializing new decoder")
        # initialize new decoder_embed, decoder_pos_embed_x, decoder_pos_embed_y,
        # decoder_blocks, decoder_norm, decoder_pred
        for key in list(checkpoint_model.keys()):
            if "decoder" in key:
                print(f"Removing key {key} from pretrained checkpoint")
                del checkpoint_model[key]
                print(f"Initializing new {key}")

        # initialize new mask_token
        key = "mask_token"
        if key in checkpoint_model:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]
            print(f"Initializing new {key}")

        # load pretrained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        assert {'pos_embed_x', 'pos_embed_y.weight'}.issubset(set(msg.missing_keys))

    skip_list = []
    if args.pretrained_encoder and args.freeze_encoder:
        # partially freeze the mode:
        # freeze patch_embed
        for n, p in model.patch_embed.named_parameters():
            p.requires_grad = False
            skip_list.append(f"patch_embed.{n}")
        # freeze encoder
        for n, p in model.blocks[:].named_parameters():
            p.requires_grad = False
            skip_list.append(f"blocks.{n}")
        # freeze norm
        for n, p in model.norm.named_parameters():
            p.requires_grad = False
            skip_list.append(f"norm.{n}")

    if new_patch_size == True:
        # unfreeze patch_embed
        for n, p in model.patch_embed.named_parameters():
            p.requires_grad = True
            skip_list = [module for module in skip_list if "patch_embed" not in module]
        # unfreeze norm
        for n, p in model.norm.named_parameters():
            p.requires_grad = True
            skip_list = [module for module in skip_list if module not in ["norm.weight", "norm.bias"]]
    
    print(skip_list)

    if args.compile:
        model.forward = torch.compile(model.forward, dynamic=True)
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

    if args.min_lr is None:
        args.min_lr = args.lr * 0.1

    print("base lr: %.2e" % (args.lr * 32 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay_timm_lrd(model_without_ddp, args.weight_decay, layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler(amp_dtype=args.amp_dtype)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")
    
    best_stats = {'epoch':-1, 'total_loss':np.inf, 'loss':np.inf, 'ncc':0.0, 'cos_sim':-1.0, 'mse':np.inf, 'mae':np.inf, 'patch_patch_sim':np.inf, 'rankme':0.0, 'kshot_cls_acc':0.0}
    best_eval_scores = {'count':1, 'nb_ckpts_max':3, 'eval_criterion':[best_stats[args.eval_criterion]]}

    if args.eval_criterion == 'kshot_cls_acc' and not (args.online_kshot and args.online_kshot_data_base):
        raise ValueError("--eval_criterion kshot_cls_acc requires --online_kshot with --online_kshot_data_base")
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        if True: #args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats, train_history = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler,
                                                     log_writer=log_writer, args=args)

        eval_stats = train_stats
        if args.val_data_path:
            val_stats, val_history = evaluate(data_loader_val, model, device, epoch, 
                                              log_writer=log_writer, args=args)
            eval_stats = val_stats

        # Fig.1 diagnostics — every 2 epochs.
        online_diag_history = {}
        if misc.is_main_process() and (epoch % 2 == 0 or epoch == args.epochs - 1):
            if args.online_kshot and args.online_kshot_data_base:
                try:
                    online_diag_history.update(
                        evaluate_online_kshot(model_without_ddp, device, args.online_kshot_data_base))
                except Exception as e:
                    print(f"[online_kshot] skipped due to {type(e).__name__}: {e}")
            if args.online_forecast and args.online_forecast_data_path:
                try:
                    online_diag_history.update(
                        evaluate_online_forecast(model_without_ddp, device,
                                                 args.online_forecast_data_path, args,
                                                 mask_ratio=args.online_forecast_mask_ratio,
                                                 time_steps=args.online_forecast_time_steps))
                except Exception as e:
                    print(f"[online_forecast] skipped due to {type(e).__name__}: {e}")

        # Expose the online k-shot CLS balanced accuracy as a selectable eval metric.
        # Absent on epochs where k-shot didn't run; the selection block below skips this epoch in that case.
        if 'online/kshot_cls_avg_acc_balanced' in online_diag_history:
            eval_stats['kshot_cls_acc'] = online_diag_history['online/kshot_cls_avg_acc_balanced']

        # Per-layer patch-token L2 norm density panels.
        token_norm_history = {}
        if (args.track_token_norms and misc.is_main_process()
                and (epoch % args.track_token_norms_freq == 0 or epoch == args.epochs - 1)):
            try:
                enc_norms, dec_norms, enc_dim, dec_dim = track_token_l2_norms(
                    model, data_loader_token_norms, device,
                    num_samples=args.track_token_norms_samples, mode="both")
                img = plot_token_l2_norms(encoder_norms=enc_norms, decoder_norms=dec_norms,
                                          encoder_embed_dim=enc_dim, decoder_embed_dim=dec_dim)
                if img is not None:
                    token_norm_history["Token L2 Norms"] = img
                # Record the last-block, pre-final-norm distribution (index -2;
                # index -1 is post-LayerNorm, which flattens the outlier tail).
                if enc_norms is not None and len(enc_norms) >= 2:
                    last_layer_norm_history_enc.append((epoch, enc_norms[-2].clone()))
                    last_layer_norm_enc_dim = enc_dim
                if dec_norms is not None and len(dec_norms) >= 2:
                    last_layer_norm_history_dec.append((epoch, dec_norms[-2].clone()))
                    last_layer_norm_dec_dim = dec_dim

                # Stash the full per-layer norms at the final epoch so the
                # "Token L2 Norms" figure can be regenerated offline.
                if epoch == args.epochs - 1:
                    final_token_norms_enc = (
                        [t.clone() for t in enc_norms] if enc_norms is not None else None)
                    final_token_norms_dec = (
                        [t.clone() for t in dec_norms] if dec_norms is not None else None)
                    final_token_norms_epoch = epoch

                # Persist the diagnostic to disk every time it runs, so partial
                # runs (early stop / crash) keep their data. File is rewritten
                # in full each call (the file is small).
                if args.output_dir and (last_layer_norm_history_enc or last_layer_norm_history_dec):
                    torch.save({
                        "encoder": last_layer_norm_history_enc,
                        "decoder": last_layer_norm_history_dec,
                        "encoder_embed_dim": last_layer_norm_enc_dim,
                        "decoder_embed_dim": last_layer_norm_dec_dim,
                    }, os.path.join(args.output_dir, "last_layer_norms.pt"))
                if args.output_dir and (final_token_norms_enc is not None
                                        or final_token_norms_dec is not None):
                    torch.save({
                        "encoder": final_token_norms_enc,
                        "decoder": final_token_norms_dec,
                        "encoder_embed_dim": last_layer_norm_enc_dim,
                        "decoder_embed_dim": last_layer_norm_dec_dim,
                        "epoch": final_token_norms_epoch,
                    }, os.path.join(args.output_dir, "token_norms_final.pt"))
            except Exception as e:
                print(f"[track_token_norms] skipped due to {type(e).__name__}: {e}")

        # online evaluation of the downstream task
        online_history = {}
        if args.online_evaluation and epoch % 10 == 0:
            if args.online_evaluation_task == "classification":
                online_estimator = LogisticRegression(class_weight='balanced', max_iter=2000)
            elif args.online_evaluation_task == "regression":
                online_estimator = LinearRegression()
            
            online_history = evaluate_online(estimator=online_estimator, model=model_without_ddp, device=device, 
                                             train_dataloader=data_loader_online_train, 
                                             val_dataloader=data_loader_online_val, args=args)
        
        if args.eval_criterion == "epoch":
            best_stats['epoch'] = epoch
            if args.output_dir:
                # save the best nb_ckpts_max checkpoints
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'], reverse=True)
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(epoch)

                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, nb_ckpts_max=best_eval_scores['nb_ckpts_max'], 
                    domains=dataset_train.domains, domain_offsets=dataset_train.offsets)
        elif args.eval_criterion in ["total_loss", "loss", "mse", "mae", "patch_patch_sim"]:
            if early_stop.evaluate_decreasing_metric(val_metric=eval_stats[args.eval_criterion]) and misc.is_main_process():
                print("Early stopping the training")
                break

            if args.output_dir and eval_stats[args.eval_criterion] <= max(best_eval_scores['eval_criterion']):
                # save the best nb_ckpts_max checkpoints
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'])
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(eval_stats[args.eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=eval_stats, 
                    evaluation_criterion=args.eval_criterion, nb_ckpts_max=best_eval_scores['nb_ckpts_max'], 
                    mode="decreasing", domains=dataset_train.domains, domain_offsets=dataset_train.offsets)
        elif args.eval_criterion not in eval_stats:
            # Criterion not computed this epoch (e.g. kshot_cls_acc only runs every
            # other epoch). Skip checkpointing / early-stop for this step.
            print(f"[eval] skipping checkpoint: '{args.eval_criterion}' not available this epoch")
        else:
            if early_stop.evaluate_increasing_metric(val_metric=eval_stats[args.eval_criterion]) and misc.is_main_process():
                print("Early stopping the training")
                break

            if args.output_dir and eval_stats[args.eval_criterion] >= min(best_eval_scores['eval_criterion']):
                # save the best nb_ckpts_max checkpoints
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'], reverse=True)
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(eval_stats[args.eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=eval_stats,
                    evaluation_criterion=args.eval_criterion, nb_ckpts_max=best_eval_scores['nb_ckpts_max'],
                    mode="increasing", domains=dataset_train.domains, domain_offsets=dataset_train.offsets)

        # always persist the final-epoch checkpoint, independent of eval_criterion
        if args.output_dir and epoch == args.epochs - 1 and args.eval_criterion != "epoch":
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, nb_ckpts_max=best_eval_scores['nb_ckpts_max'],
                domains=dataset_train.domains, domain_offsets=dataset_train.offsets)

        best_stats['total_loss'] = min(best_stats['total_loss'], eval_stats['total_loss'])
        best_stats['loss'] = min(best_stats['loss'], eval_stats['loss'])
        best_stats['ncc'] = max(best_stats['ncc'], eval_stats['ncc'])
        best_stats['cos_sim'] = max(best_stats['cos_sim'], eval_stats['cos_sim'])
        best_stats['mse'] = min(best_stats['mse'], eval_stats['mse'])
        best_stats['mae'] = min(best_stats['mae'], eval_stats['mae'])
        if 'patch_patch_sim' in eval_stats:
            best_stats['patch_patch_sim'] = min(best_stats['patch_patch_sim'], eval_stats['patch_patch_sim'])
        if 'rankme' in eval_stats:
            best_stats['rankme'] = max(best_stats['rankme'], eval_stats['rankme'])
        if 'kshot_cls_acc' in eval_stats:
            best_stats['kshot_cls_acc'] = max(best_stats['kshot_cls_acc'], eval_stats['kshot_cls_acc'])

        print(f"Total Loss / Loss / Normalized Cross-Correlation (NCC) / Cosine Similarity / Mean Squared Error (MSE) / ",
              f"Mean Absolute Error (MAE) of the network on {len((dataset_val if args.val_data_path else dataset_train))} val samples: {eval_stats['total_loss']:.4f} / ",
              f"{eval_stats['loss']:.4f} / {eval_stats['ncc']:.2f} / {eval_stats['cos_sim']:.2f} / {eval_stats['mse']:.2f} / ",
              f"{eval_stats['mae']:.2f}")

        print(f"Min Total Loss / Min Loss / Max NCC / Max Cosine Similarity / Min MSE / Min MAE: ",
              f"{best_stats['total_loss']:.4f} / {best_stats['loss']:.4f} / {best_stats['ncc']:.2f} / ", 
              f"{best_stats['cos_sim']:.2f} / {best_stats['mse']:.2f} / {best_stats['mae']:.2f}\n")
        
        total_time = time.time() - start_time
        # online_history / online_diag_history may contain wandb.Image objects
        # (Reconstruction, sine plots, …) that are not JSON-serialisable; only
        # carry the scalar entries into log.txt.
        def _scalars(d):
            return {k: v for k, v in d.items()
                    if isinstance(v, (int, float, bool, str)) or v is None}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **({f'val_{k}': v for k, v in val_stats.items()} if args.val_data_path else {}),
                     **_scalars(online_history),
                     **_scalars(online_diag_history),
                     'n_parameters': n_parameters,
                     'epoch': epoch,
                     'time_per_epoch' : total_time}
        
        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if args.wandb and misc.is_main_process():
            log_data = {**train_history,
                        **(val_history if args.val_data_path else {}),
                        **online_history,
                        **online_diag_history,
                        **token_norm_history,
                        "Time per epoch [sec]": total_time}
            wandb.log(log_data)

    # End-of-training: render the last-block patch-token L2 norm evolution
    # (à la Darcet et al. 2024, Fig 4(b)). Raw history was persisted to disk
    # incrementally inside the per-epoch diagnostic block.
    if (last_layer_norm_history_enc or last_layer_norm_history_dec) and misc.is_main_process():
        if args.wandb:
            evo_img = plot_last_layer_norm_evolution(
                encoder_history=last_layer_norm_history_enc or None,
                decoder_history=last_layer_norm_history_dec or None,
                encoder_embed_dim=last_layer_norm_enc_dim,
                decoder_embed_dim=last_layer_norm_dec_dim)
            if evo_img is not None:
                wandb.log({"Last-layer token L2 norm evolution": evo_img})

    if args.wandb and misc.is_main_process():
        wandb.log({f'Best Statistics/{k}': v for k, v in best_stats.items()})
        wandb.finish()
        exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)