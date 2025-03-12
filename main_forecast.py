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
from typing import Tuple
import numpy as np
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import wandb
# os.environ["WANDB__SERVICE_WAIT"] = "500"

# assert timm.__version__ == "0.3.2"  # version check
# import timm.optim.optim_factory as optim_factory

from util.dataset import TimeSeriesDataset
import util.misc as misc
from util.misc import add_weight_decay
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed_x, interpolate_decoder_pos_embed_x
from util.callbacks import EarlyStop

import models_otis

from engine_pretrain import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('OTiS generative finetuning', add_help=False)
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
    
    output_projections = ["decoder", "mlp"]
    parser.add_argument('--output_projection', default='decoder', type=str, choices=output_projections,
                        help='Projection of the masked tokens (default: decoder)')

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
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

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
                        help='Early stopping, if val is worse than train for specified nb of epochs \
                              (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')
    
    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--ignore_pos_embed_y', action='store_true', default=False,
                        help='Ignore pre-trained position embeddings Y (spatial axis) from checkpoint')
    
    parser.add_argument('--zero_shot', action='store_true', default=False,
                        help='Freeze all parameters except for the position embeddings Y (spatial axis) of the encoder')
    
    parser.add_argument('--freeze_pos_embed_y', action='store_true', default=False,
                        help='Freeze position embeddings Y (spatial axis) of the encoder')
    encoder_modules = ["none", "all", "ffn", "attn"]
    parser.add_argument('--freeze_encoder', default='none', type=str, choices=encoder_modules,
                        help='Freeze encoder modules (default: none)')
    
    parser.add_argument('--freeze_mask_token', action='store_true', default=False,
                        help='Freeze mask token')
    parser.add_argument('--freeze_decoder_pos_embed_y', action='store_true', default=False,
                        help='Freeze position embeddings Y (spatial axis) of the decoder')
    parser.add_argument('--freeze_decoder', action='store_true', default=False,
                        help='Freeze all decoder modules')

    # Dataset parameters
    downstream_tasks = ["forecasting", "imputation"]
    parser.add_argument('--downstream_task', default='forecasting', type=str, choices=downstream_tasks,
                        help='downstream task (default: forecasting)')
    
    eval_criterions = ["epoch", "total_loss", "loss", "ncc", "mse", "mae"]
    parser.add_argument('--eval_criterion', default='mse', type=str, choices=eval_criterions,
                        help='downstream task evaluation metric (default: mse)')
    
    parser.add_argument('--data_path', default='_.pt', type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default='', type=str,
                        help='validation dataset path')

    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_path', default='', type=str,
                        help='test dataset path (default: None)')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log (default: ./logs)')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_entity', default='', type=str,
                        help='entity of the current run')
    parser.add_argument('--wandb_project', default='',
                        help='project where to wandb log')
    parser.add_argument('--wandb_id', default='', type=str,
                        help='id of the current run')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--save_embeddings', action='store_true', default=False,
                        help='save model embeddings')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=0, type=int)
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

    # wandb logging
    if args.wandb == True and misc.is_main_process():
        config = vars(args)
        if args.wandb_id:
            wandb.init(project=args.wandb_project, id=args.wandb_id, config=config, entity=args.wandb_entity)
        else:
            wandb.init(project=args.wandb_project, config=config, entity=args.wandb_entity)

        args.__dict__ = wandb.config.as_dict()

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
    dataset_train = TimeSeriesDataset(data_path=args.data_path, 
                                      univariate=args.univariate,
                                      train=True, 
                                      N_val=args.batch_size,
                                      args=args)
    dataset_val = TimeSeriesDataset(data_path=args.val_data_path, 
                                    domain_offsets=dataset_train.offsets, 
                                    univariate=args.univariate,
                                    train=False, 
                                    N_val=-1,
                                    args=args)
    if args.test:
        dataset_test = TimeSeriesDataset(data_path=args.test_data_path, 
                                        domain_offsets=dataset_train.offsets, 
                                        univariate=args.univariate,
                                        train=False, 
                                        test=True,
                                        N_val=-1,
                                        args=args)

    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))
    if args.test:
        print("Test set size: ", len(dataset_test))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        print(f"num_tasks: {num_tasks}")
        global_rank = misc.get_rank()
        print(f"global_rank: {global_rank}")
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        # print("Sampler_train = %s" % str(sampler_train))

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

        if args.test:
            if args.dist_eval:
                if len(dataset_test) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)  # shuffle=True to reduce monitor bias
            else:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if args.test:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # tensorboard logging
    if False: #global_rank == 0 and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

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

    if args.test:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            sampler=sampler_test,
            # shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset_test.collate_fn,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    # define the model
    model = models_otis.__dict__[args.model](
        domains=dataset_train.domains,
        domain_weights=dataset_train.domain_weights,
        input_channels=args.input_channels,
        time_steps=args.time_steps,
        patch_size=args.patch_size,
        output_projection=args.output_projection,
        separate_dec_pos_embed_y=args.separate_dec_pos_embed_y,
        norm_pix_loss=args.norm_pix_loss,
        masked_patch_loss=args.masked_patch_loss,
        domain_weighted_loss=args.domain_weighted_loss,
        downstream=args.downstream_task
    )

    new_patch_size = False
    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load pretrained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']

        # check if new and old patch_size match
        nb_channels_ckpt = checkpoint_model['patch_embed.proj.weight'].shape[-3]
        nb_channels_model = args.input_size[0]

        checkpoint_patch_size = checkpoint_model['patch_embed.proj.weight'].shape[-2:]
        patch_height_ckpt, patch_width_ckpt = checkpoint_patch_size[0], checkpoint_patch_size[1]
        patch_height_model, patch_width_model = args.patch_size[0], args.patch_size[1]

        if nb_channels_ckpt != nb_channels_model or patch_height_ckpt != patch_height_model or patch_width_ckpt != patch_width_model:
            new_patch_size = True
            # initialize new patch_embed
            for key in ["patch_embed.proj.weight", "patch_embed.proj.bias", 
                        "patch_embed.norm.weight", "patch_embed.norm.bias"]:
                if key in checkpoint_model:
                    print(f"Removing key {key} from pretrained checkpoint")
                    del checkpoint_model[key]
            print("Initializing new patch_embed")

            # initialize new decoder_pred
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
        print(f"Identified domain: {dataset_train.domains}")
        assert len(dataset_train.domains) == 1, "There is more than one domain in the target dataset"
        target_domain = list(dataset_train.domains.keys())[0]
        # target_shape = list(dataset_train.domains.values())[0]

        pos_embed_y_available = False
        checkpoint_domains = checkpoint["domains"]
        for domain, shape in checkpoint_domains.items():
            if domain == target_domain: # and shape[1] == target_shape[1]:
                pos_embed_y_available = True
                break

        if len(checkpoint["domain_offsets"]) > 1 and sum([v for v in checkpoint["domain_offsets"].values()]) == 0:
            # domain-agnostic pos_embed_y
            print("INFO: Found domain-agnostic pos_embed_y in checkpoint")
            pos_embed_y_available = True # if pos_embed_y_available = False before

            # set offset to zero
            print(dataset_train.domain)
            checkpoint["domain_offsets"][dataset_train.domain[0][0]] = 0

        if not args.ignore_pos_embed_y and pos_embed_y_available:
            print("Loading pos_embed_y from checkpoint")
            print(f"Current pos_embed_y shape: {model.pos_embed_y.weight.shape}")
            model.pos_embed_y = None
            model.pos_embed_y = torch.nn.Embedding.from_pretrained(checkpoint_model["pos_embed_y.weight"])
            print(f"New pos_embed_y shape: {model.pos_embed_y.weight.shape}")

            # load domain_offsets
            dataset_train.set_domain_offsets(checkpoint["domain_offsets"])
            dataset_val.set_domain_offsets(checkpoint["domain_offsets"])
            if args.test:
                dataset_test.set_domain_offsets(checkpoint["domain_offsets"])
        else:
            print("Initializing new pos_embed_y")

        key = "pos_embed_y.weight"
        if key in checkpoint_model:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]

        if False: #args.ignore_decoder:
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
        else:
            # load decoder_pos_embed_x
            interpolate_decoder_pos_embed_x(model, checkpoint_model)

            key = "decoder_pos_embed_x"
            if key in checkpoint_model:
                print(f"Removing key {key} from pretrained checkpoint")
                del checkpoint_model[key]

        # load pretrained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        assert {'pos_embed_x', 'pos_embed_y.weight'}.issubset(set(msg.missing_keys))

    # keep track of trainable parameters
    trainable_params = []

    if args.zero_shot:
        # freeze the model
        for _, p in model.named_parameters():
            p.requires_grad = False

        # only pos_embed_y is trainable
        for n, p in model.pos_embed_y.named_parameters():
            p.requires_grad = True
            trainable_params.append(f"pos_embed_y.{n}")
    else:
        # default: entire model is trainable
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad == True]

        if args.freeze_encoder != "none":
            # freeze patch_embed
            for n, p in model.patch_embed.named_parameters():
                p.requires_grad = False
                trainable_params.remove(f"patch_embed.{n}")
            # freeze norm
            for n, p in model.norm.named_parameters():
                p.requires_grad = False
                trainable_params.remove(f"norm.{n}")

            if args.freeze_encoder == "ffn":
                # freeze the mlp and norm layers
                for n, p in model.blocks[:].named_parameters():
                    if "norm2" in n or "mlp" in n:
                        p.requires_grad = False
                        trainable_params.remove(f"blocks.{n}")
            elif args.freeze_encoder == "attn":
                # freeze the attn and norm layers
                for n, p in model.blocks[:].named_parameters():
                    if "norm1" in n or "attn" in n:
                        p.requires_grad = False
                        trainable_params.remove(f"blocks.{n}")
            else:
                # freeze the entire encoder
                for n, p in model.blocks[:].named_parameters():
                    p.requires_grad = False
                    trainable_params.remove(f"blocks.{n}")
        
        if args.freeze_pos_embed_y:
            # freeze pos_embed_y
            for n, p in model.pos_embed_y.named_parameters():
                p.requires_grad = False
                trainable_params.remove(f"pos_embed_y.{n}")

        if args.freeze_decoder:
            # freeze decoder_embed
            for n, p in model.decoder_embed.named_parameters():
                p.requires_grad = False
                trainable_params.remove(f"decoder_embed.{n}")
            # freeze decoder
            for n, p in model.decoder_blocks[:].named_parameters():
                p.requires_grad = False
                trainable_params.remove(f"decoder_blocks.{n}")
            # freeze decoder_norm
            for n, p in model.decoder_norm.named_parameters():
                p.requires_grad = False
                trainable_params.remove(f"decoder_norm.{n}")
            # freeze decoder_pred
            for n, p in model.decoder_pred.named_parameters():
                p.requires_grad = False
                trainable_params.remove(f"decoder_pred.{n}")
        
        if args.freeze_decoder_pos_embed_y:
            # freeze decoder_pos_embed_y
            for n, p in model.decoder_pos_embed_y.named_parameters():
                p.requires_grad = False
                trainable_params.remove(f"decoder_pos_embed_y.{n}")

        if args.freeze_mask_token:
            model.mask_token.requires_grad = False
            trainable_params.remove(f"mask_token")

    if new_patch_size:
        # train patch_embed
        for n, p in model.patch_embed.named_parameters():
            p.requires_grad = True
            trainable_params.append(f"patch_embed.{n}")

        # train decoder_pred
        for n, p in model.decoder_pred.named_parameters():
            p.requires_grad = True
            trainable_params.append(f"decoder_pred.{n}")

        # train mask token
        model.mask_token.requires_grad = True
        trainable_params.append(f"mask_token")

    if args.compile:
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")
    model.to(device, non_blocking=True)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_encoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "decoder" not in n)
    # params_encoder = [n for n, p in model.named_parameters() if p.requires_grad and "decoder" not in n]
    n_params_decoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "decoder" in n)
    # params_decoder = [n for n, p in model.named_parameters() if p.requires_grad and "decoder" in n]

    print("Model = %s" % str(model_without_ddp))
    print('Number of params (k): %.2f' % (n_parameters / 1.e3))
    print('Number of encoder params (k): %.2f' % (n_params_encoder / 1.e3))
    # print(params_encoder)
    print('Number of decoder params (k): %.2f' % (n_params_decoder / 1.e3))
    # print(params_decoder)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 32

    print("base lr: %.2e" % (args.lr * 32 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups_0, skip_list = [], []
    if args.finetune and not args.zero_shot:
        # increase the learning rate of the encoder's and decoder's pos_embed_y  
        skip_list = [elem for elem in trainable_params if "pos_embed_y" not in elem]
        param_groups_0, skip_list = add_weight_decay(model_without_ddp, args.weight_decay, lr_scale=10, skip_list=skip_list)

    param_groups, _ = add_weight_decay(model_without_ddp, args.weight_decay, skip_list=skip_list)
    param_groups = param_groups + param_groups_0

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if not args.eval:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        sub_strings = args.resume.split("/")
        if "checkpoint" in sub_strings[-1]:
            nb_ckpts = 1
        else:
            nb_ckpts = int(sub_strings[-1])+1

        for epoch in range(0, nb_ckpts):
            if "checkpoint" not in sub_strings[-1]:
                args.resume = "/".join(sub_strings[:-1]) + "/checkpoint-" + str(epoch) + ".pth"

            misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

            test_stats, test_history = evaluate(data_loader_val, model, device, epoch, 
                                                log_writer=log_writer, args=args)

            print(f"Mean Squared Error (MSE) / Mean Absolute Error (MAE) / Normalized Cross-Correlation (NCC)", 
                  f"of the network on {len(dataset_val)} test samples: {test_stats['mse']:.4f} / {test_stats['mae']:.4f} /", 
                  f"{test_stats['ncc']:.4f}")
        
            if args.wandb and misc.is_main_process():
                wandb.log(test_history)

        exit(0)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")
    
    best_stats = {'epoch':-1, 'total_loss':np.inf, 'loss':np.inf, 'ncc':0.0, 'cos_sim':-1.0, 'mse':np.inf, 'mae':np.inf}
    best_eval_scores = {'count':1, 'nb_ckpts_max':1, 'eval_criterion':[best_stats[args.eval_criterion]]}
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        if epoch == 0:
            print(f"Trainable parameters:\n{trainable_params}")

        if True: #args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats, train_history = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler,
                                                     log_writer=log_writer, args=args)

        val_stats, val_history = evaluate(data_loader_val, model, device, epoch,
                                          log_writer=log_writer, args=args)

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
        elif args.eval_criterion in ["total_loss", "loss", "mse", "mae"]:
            if early_stop.evaluate_decreasing_metric(val_metric=val_stats[args.eval_criterion]) and misc.is_main_process():
                print("Early stopping the training")
                break

            if args.output_dir and val_stats[args.eval_criterion] <= max(best_eval_scores['eval_criterion']):
                # save the best nb_ckpts_max checkpoints
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'])
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(val_stats[args.eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, 
                    evaluation_criterion=args.eval_criterion, nb_ckpts_max=best_eval_scores['nb_ckpts_max'], 
                    mode="decreasing", domains=dataset_train.domains, domain_offsets=dataset_train.offsets)
        else:
            if early_stop.evaluate_increasing_metric(val_metric=val_stats[args.eval_criterion]) and misc.is_main_process():
                print("Early stopping the training")
                break

            if args.output_dir and val_stats[args.eval_criterion] >= min(best_eval_scores['eval_criterion']):
                # save the best nb_ckpts_max checkpoints
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'], reverse=True)
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(val_stats[args.eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=val_stats, 
                    evaluation_criterion=args.eval_criterion, nb_ckpts_max=best_eval_scores['nb_ckpts_max'], 
                    mode="increasing", domains=dataset_train.domains, domain_offsets=dataset_train.offsets)

        best_stats['total_loss'] = min(best_stats['total_loss'], val_stats['total_loss'])
        best_stats['loss'] = min(best_stats['loss'], val_stats['loss'])
        best_stats['ncc'] = max(best_stats['ncc'], val_stats['ncc'])
        best_stats['cos_sim'] = max(best_stats['cos_sim'], val_stats['cos_sim'])
        best_stats['mse'] = min(best_stats['mse'], val_stats['mse'])
        best_stats['mae'] = min(best_stats['mae'], val_stats['mae'])

        print(f"Total Loss / Loss / Normalized Cross-Correlation (NCC) / Cosine Similarity / Mean Squared Error (MSE) / Mean Absolute Error (MAE)",
              f"of the network on {len(dataset_val)} val samples: {val_stats['total_loss']:.4f} / {val_stats['loss']:.4f} / ", 
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
            wandb.log(train_history | val_history | {"Time per epoch [sec]": total_time})

    if args.test and misc.is_main_process():
        args.resume = misc.get_best_ckpt(args.output_dir, eval_criterion=args.eval_criterion)
        
        checkpoint = torch.load(args.resume, weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Run test data on checkpoint model %s" % args.resume)
        
        test_stats, test_history = evaluate(data_loader_test, model_without_ddp, device, epoch=-1, log_writer=log_writer, args=args)
        
        actual_test_history = {}
        for k,v in test_history.items():
            key = k
            if 'val' in k:
                key = 'actual_' + key
                actual_test_history[key] = v 

        print(actual_test_history)

        if args.wandb and misc.is_main_process():
            wandb.log(actual_test_history)

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