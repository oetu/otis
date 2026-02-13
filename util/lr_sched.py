# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae?tab=readme-ov-file
# --------------------------------------------------------

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if args.min_lr == args.blr:
        return args.min_lr
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def adjust_learning_rate_wsd(optimizer, epoch, args, decay_fraction=0.1):
    """
    Warmup-Stable-Decay (WSD) learning rate schedule
    - Warmup: linearly increase the learning rate from 0 to the peak learning rate over the warmup epochs.
    - Stable: keep the learning rate constant at the peak learning rate for the epochs between warmup and decay.
    - Decay: exponential annealing (cosine) from the peak learning rate to the minimum learning rate.

    Args:
        decay_fraction: Fraction of total epochs to use for the decay phase (default: 0.1).
                        For example, with 100 total epochs and decay_fraction=0.1, the last 10 epochs will be decay.
    """
    if args.min_lr == args.blr:
        return args.min_lr

    # Calculate the epoch when decay phase starts
    decay_epochs = int(args.epochs * decay_fraction)
    decay_start_epoch = args.epochs - decay_epochs

    if epoch < args.warmup_epochs:
        # Warmup phase: linear ramp from 0 to peak LR
        lr = args.lr * epoch / args.warmup_epochs
    elif epoch < decay_start_epoch:
        # Stable phase: constant at peak LR
        lr = args.lr
    else:
        # Decay phase: cosine annealing from peak LR to min LR
        progress = (epoch - decay_start_epoch) / decay_epochs
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr