# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import random

import torch

import numpy as np

import wandb

import util.misc as misc
import util.lr_sched as lr_sched
import util.statistics as statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from sklearn.feature_selection import r_regression


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    training_history = {}

    for data_iter_step, (samples, patch_size, attn_mask, pos_embed_y, modality) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # push samples to device
        samples = samples.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        # compute model prediction
        with torch.cuda.amp.autocast():
            loss, ncc, loss_cos, cos_embed, z_std, samples_hat, samples_hat_masked = model(samples, 
                                                                                           attn_mask,
                                                                                           pos_embed_y,
                                                                                           modality,
                                                                                           mask_ratio=args.mask_ratio)

        batch_size = len(samples)

        loss_value = loss.item()
        loss_cos_value = loss_cos.item()
        cos_embed_value = cos_embed.item()
        z_std_value = z_std.item()
        ncc_value = ncc.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_cos /= accum_iter

        total_loss = loss + args.cos_weight * loss_cos
        loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        metric_logger.meters['loss_cos'].update(loss_cos_value, n=batch_size)
        metric_logger.meters['cos_embed'].update(cos_embed_value, n=batch_size)
        metric_logger.meters['z_std'].update(z_std_value, n=batch_size)
        metric_logger.meters['ncc'].update(ncc_value, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # tensorboard
    if log_writer is not None:
        log_writer.add_scalar('train/train_loss', train_stats["loss"], epoch)
        log_writer.add_scalar('train/train_loss_cos', train_stats["loss_cos"], epoch)
        log_writer.add_scalar('train/train_cos_embed', train_stats["cos_embed"], epoch)
        log_writer.add_scalar('train/train_z_std', train_stats["z_std"], epoch)
        log_writer.add_scalar('lr', train_stats["lr"], epoch)
        log_writer.add_scalar('train/normalized_corr_coef', train_stats["ncc"], epoch)

    # wandb
    if args.wandb == True:
        training_history['epoch'] = epoch
        training_history['train_loss'] = train_stats["loss"]
        training_history['train_loss_cos'] = train_stats["loss_cos"]
        training_history['train_cos_embed'] = train_stats["cos_embed"]
        training_history['train_z_std'] = train_stats["z_std"]
        training_history['lr'] = train_stats["lr"]
        training_history['Normalized Correlation Coefficient'] = train_stats["ncc"]

        if (epoch % 10) == 0:
            steps = 1
            idx = random.randint(0, len(samples)-1)
            
            # (B, 1, C, T)
            attn_mask_input_space = torch.nn.functional.interpolate(attn_mask.unsqueeze(1), 
                                                                    scale_factor=patch_size, 
                                                                    mode="nearest")

            # T_indie
            max_steps = int(attn_mask_input_space[idx, 0, 0, :].sum())

            x = samples[idx][..., :max_steps:steps].detach().cpu().numpy()
            x_hat = samples_hat[idx][..., :max_steps:steps].detach().cpu().numpy()
            x_hat_masked = samples_hat_masked[idx][..., :max_steps:steps].detach().cpu().numpy()

            # samples of shape (Batch, 1, Channel, Time)
            if samples.shape[1] > 1:
                ch_idx = 2
            else:
                ch_idx = 0

            plt.close('all')
            plt.subplot(611)
            plt.plot(range(0, x.shape[-1], 1), x[0, 0, :])
            plt.subplot(612)
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, 0, :])
            plt.subplot(613)
            plt.plot(range(0, x.shape[-1], 1), x_hat_masked[0, 0, :])
            plt.subplot(614)
            plt.plot(range(0, x.shape[-1], 1), x[ch_idx, 5, :])
            plt.subplot(615)
            plt.plot(range(0, x.shape[-1], 1), x_hat[ch_idx, 5, :])
            plt.subplot(616)
            plt.plot(range(0, x.shape[-1], 1), x_hat_masked[ch_idx, 5, :])
            plt.tight_layout()
            training_history["Reconstruction"] = wandb.Image(plt)

    return train_stats, training_history

@torch.no_grad()
def evaluate_online(estimator, model, device, train_dataloader, val_dataloader, args=None):
    # switch to evaluation mode
    model.eval()

    online_history = {}
    
    # training
    train_embeddings = []
    train_labels = []
    for data, label, label_mask, pos_embed_y in train_dataloader:
        data = data.to(device, non_blocking=True)
        label = label * label_mask
        train_labels.append(label.to(device, non_blocking=True))
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            train_embeddings.append(model.forward_encoder_all_patches(data, pos_embed_y))

    train_embeddings = torch.cat(train_embeddings, dim=0)[:, 1:, :].mean(dim=1) # globally average pooled token
    train_embeddings = train_embeddings.cpu()
    train_labels = torch.cat(train_labels, dim=0)
    train_labels = train_labels.cpu()

    estimator.fit(train_embeddings, train_labels) # only fit with training data
    
    if args.online_evaluation_task == "classification":
        train_probs = torch.tensor(estimator.predict_proba(train_embeddings), dtype=torch.float16)
        classifier_f1_train = f1_score(y_true=train_labels, y_pred=train_probs.argmax(dim=-1), pos_label=1)
        classifier_acc_train = accuracy_score(y_true=train_labels, y_pred=train_probs.argmax(dim=-1))
        classifier_auc_train = roc_auc_score(y_true=torch.nn.functional.one_hot(train_labels, num_classes=-1), y_score=train_probs)
        classifier_auprc_train = average_precision_score(y_true=torch.nn.functional.one_hot(train_labels, num_classes=-1), y_score=train_probs, pos_label=1)
    elif args.online_evaluation_task == "regression":
        train_preds = torch.tensor(estimator.predict(train_embeddings), dtype=torch.float16)
        classifier_rmse_train = root_mean_squared_error(train_preds, train_labels, multioutput="raw_values")
        classifier_mae_train = mean_absolute_error(train_preds, train_labels, multioutput="raw_values")
        classifier_pcc_train = np.concatenate([r_regression(train_preds[:, i].view(-1, 1), train_labels[:, i]) for i in range(train_labels.shape[-1])], axis=0)
        classifier_r2_train = np.stack([r2_score(train_labels[:, i], train_preds[:, i]) for i in range(train_labels.shape[-1])], axis=0)

    # validation
    val_embeddings = []
    val_labels = []
    for data, label, label_mask, pos_embed_y in val_dataloader:
        data = data.to(device, non_blocking=True)
        label = label * label_mask
        val_labels.append(label.to(device, non_blocking=True))
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            val_embeddings.append(model.forward_encoder_all_patches(data, pos_embed_y))

    val_embeddings = torch.cat(val_embeddings, dim=0)[:, 1:, :].mean(dim=1) # globally average pooled token
    val_embeddings = val_embeddings.cpu()
    val_labels = torch.cat(val_labels, dim=0)
    val_labels = val_labels.cpu()
    
    if args.online_evaluation_task == "classification":
        val_probs = torch.tensor(estimator.predict_proba(val_embeddings), dtype=torch.float16)
        classifier_f1_val = f1_score(y_true=val_labels, y_pred=val_probs.argmax(dim=-1), pos_label=1)
        classifier_acc_val = accuracy_score(y_true=val_labels, y_pred=val_probs.argmax(dim=-1))
        classifier_auc_val = roc_auc_score(y_true=torch.nn.functional.one_hot(val_labels, num_classes=-1), y_score=val_probs)
        classifier_auprc_val = average_precision_score(y_true=torch.nn.functional.one_hot(val_labels, num_classes=-1), y_score=val_probs, pos_label=1)
    elif args.online_evaluation_task == "regression":
        val_preds = torch.tensor(estimator.predict(val_embeddings), dtype=torch.float16)
        classifier_rmse_val = root_mean_squared_error(val_preds, val_labels, multioutput="raw_values")
        classifier_mae_val = mean_absolute_error(val_preds, val_labels, multioutput="raw_values")
        classifier_pcc_val = np.concatenate([r_regression(val_preds[:, i].view(-1, 1), val_labels[:, i]) for i in range(val_labels.shape[-1])], axis=0)
        classifier_r2_val = np.stack([r2_score(val_labels[:, i], val_preds[:, i]) for i in range(val_labels.shape[-1])], axis=0)

    # stats
    if args.online_evaluation_task == "classification":
        online_history['online/train_f1'] = classifier_f1_train
        online_history['online/train_acc'] = classifier_acc_train
        online_history['online/train_auc'] = classifier_auc_train
        online_history['online/train_auprc'] = classifier_auprc_train

        online_history['online/val_f1'] = classifier_f1_val
        online_history['online/val_acc'] = classifier_acc_val
        online_history['online/val_auc'] = classifier_auc_val
        online_history['online/val_auprc'] = classifier_auprc_val
    elif args.online_evaluation_task == "regression":
        online_history['online/train_rmse'] = classifier_rmse_train.mean(axis=-1)
        online_history['online/train_mae'] = classifier_mae_train.mean(axis=-1)
        online_history['online/train_pcc'] = classifier_pcc_train.mean(axis=-1)
        online_history['online/train_r2'] = classifier_r2_train.mean(axis=-1)

        online_history['online/val_rmse'] = classifier_rmse_val.mean(axis=-1)
        online_history['online/val_mae'] = classifier_mae_val.mean(axis=-1)
        online_history['online/val_pcc'] = classifier_pcc_val.mean(axis=-1)
        online_history['online/val_r2'] = classifier_r2_val.mean(axis=-1)

    return online_history

@torch.no_grad()
def evaluate(data_loader, model, device, epoch, log_writer=None, args=None):
    # switch to evaluation mode
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    test_history = {}  

    for batch in metric_logger.log_every(data_loader, 10, header):
        # push samples to device
        samples = batch[0]
        samples = samples.to(device, non_blocking=True)
        
        attn_mask = batch[2]
        attn_mask = attn_mask.to(device, non_blocking=True)

        pos_embed_y = batch[3]
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        modality = batch[4]

        with torch.cuda.amp.autocast():
            loss, ncc, loss_cos, cos_embed, z_std, _, _ = model(samples,
                                                                attn_mask,
                                                                pos_embed_y,
                                                                modality,
                                                                mask_ratio=args.mask_ratio)

        batch_size = len(samples)

        loss_value = loss.item()
        loss_cos_value = loss_cos.item()
        cos_embed_value = cos_embed.item()
        z_std_value = z_std.item()
        ncc_value = ncc.item()
        
        metric_logger.update(loss=loss_value)

        metric_logger.meters['loss_cos'].update(loss_cos_value, n=batch_size)
        metric_logger.meters['cos_embed'].update(cos_embed_value, n=batch_size)
        metric_logger.meters['z_std'].update(z_std_value, n=batch_size)
        metric_logger.meters['ncc'].update(ncc_value, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # tensorboard
    if log_writer is not None:
        log_writer.add_scalar('val/val_loss', test_stats["loss"], epoch)
        log_writer.add_scalar('val/val_loss_cos', test_stats["loss_cos"], epoch)
        log_writer.add_scalar('val/val_cos_embed', test_stats["cos_embed"], epoch)
        log_writer.add_scalar('val/val_z_std', test_stats["z_std"], epoch)
        log_writer.add_scalar('val/val_normalized_corr_coef', test_stats["ncc"], epoch)

    # wandb
    if args.wandb == True:
        test_history['epoch'] = epoch
        test_history['val_loss'] = test_stats["loss"]
        test_history['val_loss_cos'] = test_stats["loss_cos"]
        test_history['val_cos_embed'] = test_stats["cos_embed"]
        test_history['val_z_std'] = test_stats["z_std"]
        test_history['Val Normalized Correlation Coefficient'] = test_stats["ncc"]

    return test_stats, test_history