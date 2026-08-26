# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae?tab=readme-ov-file
# --------------------------------------------------------

import math


def _set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_learning_rate(optimizer, step, total_steps, args, warmup_fraction=0.1):
    """
    Cosine learning rate schedule, computed across optimizer steps.
    - Warmup: linearly increase the learning rate from 0 to the peak learning rate.
    - Decay:  half-cycle cosine annealing from the peak to the minimum learning rate.

    Args:
        step:            current global optimizer step.
        total_steps:     total number of optimizer steps of the whole run.
        warmup_fraction: fraction of total steps used for warmup (default: 0.1).
    """
    if args.min_lr == args.blr:
        return args.min_lr

    warmup_steps = int(total_steps * warmup_fraction)

    if step < warmup_steps:
        # Warmup phase: linear ramp from 0 to peak LR
        lr = args.lr * step / max(warmup_steps, 1)
    else:
        # Decay phase: cosine annealing from peak LR to min LR
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * progress))

    return _set_lr(optimizer, lr)


def adjust_learning_rate_wsd(optimizer, step, total_steps, args, warmup_fraction=0.1, decay_fraction=0.1):
    """
    Warmup-Stable-Decay (WSD) learning rate schedule, computed across optimizer steps.
    - Warmup: linearly increase the learning rate from 0 to the peak learning rate.
    - Stable: keep the learning rate constant at the peak learning rate.
    - Decay:  exponential annealing (cosine) from the peak to the minimum learning rate.

    Args:
        step:            current global optimizer step.
        total_steps:     total number of optimizer steps of the whole run.
        warmup_fraction: fraction of total steps used for warmup (default: 0.1).
        decay_fraction:  fraction of total steps used for decay (default: 0.1).
                         For example, with 1000 total steps and decay_fraction=0.1,
                         the last 100 steps will be decay.
    """
    if args.min_lr == args.blr:
        return args.min_lr

    warmup_steps = int(total_steps * warmup_fraction)
    decay_steps = int(total_steps * decay_fraction)
    decay_start_step = total_steps - decay_steps

    if step < warmup_steps:
        # Warmup phase: linear ramp from 0 to peak LR
        lr = args.lr * step / max(warmup_steps, 1)
    elif step < decay_start_step:
        # Stable phase: constant at peak LR
        lr = args.lr
    else:
        # Decay phase: cosine annealing from peak LR to min LR
        progress = (step - decay_start_step) / max(decay_steps, 1)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * progress))

    return _set_lr(optimizer, lr)


def adjust_learning_rate_schedule(optimizer, step, total_steps, args):
    """
    Dispatch to the learning rate schedule selected via ``--lr_schedule``
    ('wsd', the default, or 'cosine'). Both schedules are computed across
    optimizer steps and share ``--warmup_fraction``.
    """
    warmup_fraction = getattr(args, 'warmup_fraction', 0.1)

    if getattr(args, 'lr_schedule', 'wsd') == 'cosine':
        return adjust_learning_rate(optimizer, step, total_steps, args,
                                    warmup_fraction=warmup_fraction)

    return adjust_learning_rate_wsd(optimizer, step, total_steps, args,
                                    warmup_fraction=warmup_fraction,
                                    decay_fraction=getattr(args, 'decay_fraction', 0.1))
