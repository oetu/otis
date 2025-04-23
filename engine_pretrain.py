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
matplotlib.use('Agg')           # prevents tkinter error
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
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

    for data_iter_step, (samples, attn_mask, pos_embed_y, domain) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # push samples to device
        samples = samples.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        # compute model prediction
        with torch.amp.autocast(device_type="cuda"):
            loss, ncc, cos_sim, cos_sim_embed, z_std, samples_hat, mask, _ = model(samples, 
                                                                                   attn_mask, 
                                                                                   pos_embed_y, 
                                                                                   domain, 
                                                                                   mask_ratio=args.mask_ratio)

        batch_size = len(samples)

        loss_value = loss.item()
        ncc_value = ncc.item()
        cos_sim_value = cos_sim.item()
        cos_sim_embed_value = cos_sim_embed.item()
        z_std_value = z_std.item()

        if not math.isfinite(loss_value) and misc.is_main_process():
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        ncc /= accum_iter
        cos_sim /= accum_iter

        total_loss = loss + args.ncc_weight * (1 - ncc) + args.cos_weight * cos_sim
        loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        metric_logger.meters['total_loss'].update(total_loss.item(), n=batch_size)
        
        metric_logger.meters['ncc'].update(ncc_value, n=batch_size)
        metric_logger.meters['cos_sim'].update(cos_sim_value, n=batch_size)
        metric_logger.meters['cos_sim_embed'].update(cos_sim_embed_value, n=batch_size)
        metric_logger.meters['z_std'].update(z_std_value, n=batch_size)

        # compute MSE and MAE only of the masked patches
        # (B, 1, C, T)
        # 0 is padding, 1 is actual value
        attn_mask_input_space = torch.nn.functional.interpolate(attn_mask.unsqueeze(1), 
                                                                scale_factor=args.patch_size, 
                                                                mode="nearest")

        # (B, 1, C, T)
        # 0 is keep, 1 is remove
        mask_input_space = torch.nn.functional.interpolate(mask.reshape(attn_mask.shape).unsqueeze(1), 
                                                           scale_factor=args.patch_size, 
                                                           mode="nearest")

        # (B, 1, C, T)
        combined_mask = attn_mask_input_space * mask_input_space

        # (B, 1, C, T)
        samples_diff = samples - samples_hat

        # evaluation only on the masked patches
        mse = ((samples_diff**2) * combined_mask).sum() / (combined_mask.sum() + 1e-9)
        mae = (abs(samples_diff) * combined_mask).sum() / (combined_mask.sum() + 1e-9)
        
        mse_value = mse.item()
        mae_value = mae.item()

        metric_logger.meters['mse'].update(mse_value, n=batch_size)
        metric_logger.meters['mae'].update(mae_value, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # tensorboard
    if log_writer is not None:
        log_writer.add_scalar('train/train_total_loss', train_stats["total_loss"], epoch)
        log_writer.add_scalar('train/train_loss', train_stats["loss"], epoch)
        log_writer.add_scalar('train/train_ncc', train_stats["ncc"], epoch)
        log_writer.add_scalar('train/train_cos_sim', train_stats["cos_sim"], epoch)
        log_writer.add_scalar('train/train_cos_sim_embed', train_stats["cos_sim_embed"], epoch)
        log_writer.add_scalar('train/train_z_std', train_stats["z_std"], epoch)
        log_writer.add_scalar('lr', train_stats["lr"], epoch)
        # evaluation only on the masked patches
        log_writer.add_scalar('train/train_mse', train_stats["mse"], epoch)
        log_writer.add_scalar('train/train_mae', train_stats["mae"], epoch)

    # wandb
    if args.wandb == True:
        training_history['epoch'] = epoch
        training_history['train_total_loss'] = train_stats["total_loss"]
        training_history['train_loss'] = train_stats["loss"]
        training_history['train_ncc'] = train_stats["ncc"]
        training_history['train_cos_sim'] = train_stats["cos_sim"]
        training_history['train_cos_sim_embed'] = train_stats["cos_sim_embed"]
        training_history['train_z_std'] = train_stats["z_std"]
        training_history['lr'] = train_stats["lr"]
        # evaluation only on the masked patches
        training_history['train_mse'] = train_stats["mse"]
        training_history['train_mae'] = train_stats["mae"]

        if (epoch % 10) == 0:
            steps = 1
            idx = random.randint(0, len(samples)-1)

            # T_indie
            max_steps = int(attn_mask_input_space[idx, 0, 0, :].sum())

            # (1, C, T)
            x = samples[idx][..., :max_steps:steps].detach().cpu().numpy()
            x_hat = samples_hat[idx][..., :max_steps:steps].detach().cpu().numpy()
            x_hat_masked = (samples_hat[idx] * combined_mask[idx])[..., :max_steps:steps].detach().cpu().numpy()

            ncc_0 = statistics.ncc(samples[idx, 0, 0], samples_hat[idx, 0, 0])
            ncc_0_maskedOnly = statistics.ncc(samples[idx, 0, 0], samples_hat[idx, 0, 0], combined_mask[idx, 0, 0])

            mask_0 = (mask_input_space[idx, 0, 0, :max_steps:steps]==1).cpu().numpy()

            # samples of shape (Batch, 1, Channel, Time)
            max_channels = int(attn_mask_input_space[idx, 0, :, 0].sum())
            if max_channels > 1:
                ch_idx = random.randint(1, max_channels-1)
                ncc_1 = statistics.ncc(samples[idx, 0, ch_idx], samples_hat[idx, 0, ch_idx])
                ncc_1_maskedOnly = statistics.ncc(samples[idx, 0, ch_idx], samples_hat[idx, 0, ch_idx], combined_mask[idx, 0, ch_idx])
                mask_1 = (mask_input_space[idx, 0, ch_idx, :max_steps:steps]==1).cpu().numpy()
            else:
                ch_idx = 0
                ncc_1 = ncc_0
                ncc_1_maskedOnly = ncc_0_maskedOnly
                mask_1 = mask_0

            # Plot reconstructed time series
            plt.close('all')
            plt.figure(figsize=(8, 8))

            plt.subplot(811)
            plt.title(f"Input ({domain[idx]}, channel {0})")
            plt.plot(range(0, x.shape[-1], 1), x[0, 0, :], color='black')

            plt.subplot(812)
            plt.title(f"Input vs Reconstruction (NCC {ncc_0.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x[0, 0, :], color='black')
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=min(x[0, 0, :].min(), x_hat[0, 0, :].min()), 
                             y2=max(x[0, 0, :].max(), x_hat[0, 0, :].max()), 
                             where=mask_0, color='gray', alpha=0.25)
            
            plt.subplot(813)
            plt.title(f"Reconstruction (NCC {ncc_0.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat[0, 0, :].min(), 
                             y2=x_hat[0, 0, :].max(), 
                             where=mask_0, color='gray', alpha=0.25)

            plt.subplot(814)
            plt.title(f"Reconstruction of masked patches (NCC {ncc_0_maskedOnly.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat_masked[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat_masked[0, 0, :].min(), 
                             y2=x_hat_masked[0, 0, :].max(), 
                             where=mask_0, color='gray', alpha=0.25)
            
            indices_visible_patches = np.where(mask_0 == False)
            plt.scatter(indices_visible_patches, 
                        x_hat_masked[0, 0, :][indices_visible_patches], 
                        color='white', s=7, zorder=2)

            plt.subplot(815)
            plt.title(f"Input ({domain[idx]}, channel {ch_idx})")
            plt.plot(range(0, x.shape[-1], 1), x[0, ch_idx, :], color='black')

            plt.subplot(816)
            plt.title(f"Input vs Reconstruction (NCC {ncc_1.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x[0, ch_idx, :], color='black')
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=min(x[0, ch_idx, :].min(), x_hat[0, ch_idx, :].min()), 
                             y2=max(x[0, ch_idx, :].max(), x_hat[0, ch_idx, :].max()), 
                             where=mask_1, color='gray', alpha=0.25)

            plt.subplot(817)
            plt.title(f"Reconstruction (NCC {ncc_1.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat[0, ch_idx, :].min(), 
                             y2=x_hat[0, ch_idx, :].max(), 
                             where=mask_1, color='gray', alpha=0.25)

            plt.subplot(818)
            plt.title(f"Reconstruction of masked patches (NCC {ncc_1_maskedOnly.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat_masked[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat_masked[0, ch_idx, :].min(), 
                             y2=x_hat_masked[0, ch_idx, :].max(), 
                             where=mask_1, color='gray', alpha=0.25)
            
            indices_visible_patches = np.where(mask_1 == False)
            plt.scatter(indices_visible_patches, 
                        x_hat_masked[0, ch_idx, :][indices_visible_patches], 
                        color='white', s=7, zorder=2)

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

        with torch.amp.autocast(device_type="cuda"):
            train_embeddings.append(model.forward_encoder_all_patches(data, pos_embed_y))

    train_embeddings = torch.cat(train_embeddings, dim=0)[:, 1:, :].mean(dim=1) # globally average pooled token
    train_embeddings = train_embeddings.cpu()
    train_labels = torch.cat(train_labels, dim=0)
    train_labels = train_labels.cpu()

    estimator.fit(train_embeddings, train_labels) # only fit with training data
    
    if args.online_evaluation_task == "classification":
        train_probs = torch.tensor(estimator.predict_proba(train_embeddings), dtype=torch.float16)
        classifier_f1_train = f1_score(y_true=train_labels, y_pred=train_probs.argmax(dim=-1), average="macro")
        classifier_precision_train = precision_score(y_true=train_labels, y_pred=train_probs.argmax(dim=-1), average="macro")
        classifier_recall_train = recall_score(y_true=train_labels, y_pred=train_probs.argmax(dim=-1), average="macro")
        classifier_acc_train = accuracy_score(y_true=train_labels, y_pred=train_probs.argmax(dim=-1))
        classifier_acc_balanced_train = balanced_accuracy_score(y_true=train_labels, y_pred=train_probs.argmax(dim=-1))
        if args.online_num_classes > 2:
            classifier_auc_train = roc_auc_score(y_true=train_labels, y_score=train_probs, average="macro", multi_class="ovr")
        else:
            classifier_auc_train = roc_auc_score(y_true=train_labels, y_score=train_probs[:, 1], average="macro")
        classifier_auprc_train = average_precision_score(y_true=torch.nn.functional.one_hot(train_labels, num_classes=args.online_num_classes), y_score=train_probs, average="macro")
    elif args.online_evaluation_task == "regression":
        train_preds = torch.tensor(estimator.predict(train_embeddings), dtype=torch.float16)
        classifier_rmse_train = np.float64(root_mean_squared_error(train_preds, train_labels, multioutput="raw_values"))
        classifier_mae_train = np.float64(mean_absolute_error(train_preds, train_labels, multioutput="raw_values"))
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

        with torch.amp.autocast(device_type="cuda"):
            val_embeddings.append(model.forward_encoder_all_patches(data, pos_embed_y))

    val_embeddings = torch.cat(val_embeddings, dim=0)[:, 1:, :].mean(dim=1) # globally average pooled token
    val_embeddings = val_embeddings.cpu()
    val_labels = torch.cat(val_labels, dim=0)
    val_labels = val_labels.cpu()
    
    if args.online_evaluation_task == "classification":
        val_probs = torch.tensor(estimator.predict_proba(val_embeddings), dtype=torch.float16)
        classifier_f1_val = f1_score(y_true=val_labels, y_pred=val_probs.argmax(dim=-1), average="macro")
        classifier_precision_val = precision_score(y_true=val_labels, y_pred=val_probs.argmax(dim=-1), average="macro")
        classifier_recall_val = recall_score(y_true=val_labels, y_pred=val_probs.argmax(dim=-1), average="macro")
        classifier_acc_val = accuracy_score(y_true=val_labels, y_pred=val_probs.argmax(dim=-1))
        classifier_acc_balanced_val = balanced_accuracy_score(y_true=val_labels, y_pred=val_probs.argmax(dim=-1))
        if args.online_num_classes > 2:
            classifier_auc_val = roc_auc_score(y_true=val_labels, y_score=val_probs, average="macro", multi_class="ovr")
        else:
            classifier_auc_val = roc_auc_score(y_true=val_labels, y_score=val_probs[:, 1], average="macro")
        classifier_auprc_val = average_precision_score(y_true=torch.nn.functional.one_hot(val_labels, num_classes=args.online_num_classes), y_score=val_probs, average="macro")
    elif args.online_evaluation_task == "regression":
        val_preds = torch.tensor(estimator.predict(val_embeddings), dtype=torch.float16)
        classifier_rmse_val = np.float64(root_mean_squared_error(val_preds, val_labels, multioutput="raw_values"))
        classifier_mae_val = np.float64(mean_absolute_error(val_preds, val_labels, multioutput="raw_values"))
        classifier_pcc_val = np.concatenate([r_regression(val_preds[:, i].view(-1, 1), val_labels[:, i]) for i in range(val_labels.shape[-1])], axis=0)
        classifier_r2_val = np.stack([r2_score(val_labels[:, i], val_preds[:, i]) for i in range(val_labels.shape[-1])], axis=0)

    # stats
    if args.online_evaluation_task == "classification":
        online_history['online/train_f1'] = classifier_f1_train
        online_history['online/train_precision'] = classifier_precision_train
        online_history['online/train_recall'] = classifier_recall_train
        online_history['online/train_acc'] = classifier_acc_train
        online_history['online/train_acc_balanced'] = classifier_acc_balanced_train
        online_history['online/train_auc'] = classifier_auc_train
        online_history['online/train_auprc'] = classifier_auprc_train

        online_history['online/val_f1'] = classifier_f1_val
        online_history['online/val_precision'] = classifier_precision_val
        online_history['online/val_recall'] = classifier_recall_val
        online_history['online/val_acc'] = classifier_acc_val
        online_history['online/val_acc_balanced'] = classifier_acc_balanced_val
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
    embeddings = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        # push samples to device
        samples = batch[0]
        samples = samples.to(device, non_blocking=True)
        
        attn_mask = batch[1]
        attn_mask = attn_mask.to(device, non_blocking=True)

        pos_embed_y = batch[2]
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        domain = batch[3]

        with torch.amp.autocast(device_type="cuda"):
            loss, ncc, cos_sim, cos_sim_embed, z_std, samples_hat, mask, latent = model(samples, 
                                                                                        attn_mask, 
                                                                                        pos_embed_y, 
                                                                                        domain, 
                                                                                        mask_ratio=args.mask_ratio)

        if args.save_embeddings:
            # latent of shape (B, 1+N', D)
            embedding = latent[:, :1, :].mean(dim=1) # (B, D)
            embeddings.append(embedding)

        batch_size = len(samples)

        loss_value = loss.item()
        ncc_value = ncc.item()
        cos_sim_value = cos_sim.item()
        cos_sim_embed_value = cos_sim_embed.item()
        z_std_value = z_std.item()
        
        metric_logger.update(loss=loss_value)

        total_loss_value = loss_value + args.ncc_weight * (1 - ncc_value) + args.cos_weight * cos_sim_value
        metric_logger.meters['total_loss'].update(total_loss_value, n=batch_size)
        
        metric_logger.meters['ncc'].update(ncc_value, n=batch_size)
        metric_logger.meters['cos_sim'].update(cos_sim_value, n=batch_size)
        metric_logger.meters['cos_sim_embed'].update(cos_sim_embed_value, n=batch_size)
        metric_logger.meters['z_std'].update(z_std_value, n=batch_size)

        # compute MSE and MAE only of the masked patches
        # (B, 1, C, T)
        # 0 is padding, 1 is actual value
        attn_mask_input_space = torch.nn.functional.interpolate(attn_mask.unsqueeze(1), 
                                                                scale_factor=args.patch_size, 
                                                                mode="nearest")

        # (B, 1, C, T)
        # 0 is keep, 1 is remove
        mask_input_space = torch.nn.functional.interpolate(mask.reshape(attn_mask.shape).unsqueeze(1), 
                                                            scale_factor=args.patch_size, 
                                                            mode="nearest")

        # (B, 1, C, T)
        combined_mask = attn_mask_input_space * mask_input_space

        # (B, 1, C, T)
        samples_diff = samples - samples_hat

        # evaluation only on the masked patches
        mse = ((samples_diff**2) * combined_mask).sum() / (combined_mask.sum() + 1e-9)
        mae = (abs(samples_diff) * combined_mask).sum() / (combined_mask.sum() + 1e-9)

        mse_value = mse.item()
        mae_value = mae.item()

        metric_logger.meters['mse'].update(mse_value, n=batch_size)
        metric_logger.meters['mae'].update(mae_value, n=batch_size)

    if args.save_embeddings and misc.is_main_process():
        embeddings = torch.cat(embeddings, dim=0).to(device="cpu", dtype=torch.float32).detach() # (B, D)
        
        embeddings_path = os.path.join(args.output_dir, "embeddings")
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)
        
        file_name = f"embeddings_{epoch}.pt"
        torch.save(embeddings, os.path.join(embeddings_path, file_name))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # tensorboard
    if log_writer is not None:
        log_writer.add_scalar('val/val_total_loss', test_stats["total_loss"], epoch)
        log_writer.add_scalar('val/val_loss', test_stats["loss"], epoch)
        log_writer.add_scalar('val/val_ncc', test_stats["ncc"], epoch)
        log_writer.add_scalar('val/val_cos_sim', test_stats["cos_sim"], epoch)
        log_writer.add_scalar('val/val_cos_sim_embed', test_stats["cos_sim_embed"], epoch)
        log_writer.add_scalar('val/val_z_std', test_stats["z_std"], epoch)
        # evaluation only on the masked patches
        log_writer.add_scalar('val/val_mse', test_stats["mse"], epoch)
        log_writer.add_scalar('val/val_mae', test_stats["mae"], epoch)

    # wandb
    if args.wandb == True:
        test_history['epoch'] = epoch
        test_history['val_total_loss'] = test_stats["total_loss"]
        test_history['val_loss'] = test_stats["loss"]
        test_history['val_ncc'] = test_stats["ncc"]
        test_history['val_cos_sim'] = test_stats["cos_sim"]
        test_history['val_cos_sim_embed'] = test_stats["cos_sim_embed"]
        test_history['val_z_std'] = test_stats["z_std"]
        # evaluation only on the masked patches
        test_history['val_mse'] = test_stats["mse"]
        test_history['val_mae'] = test_stats["mae"]

        if (epoch % 10) == 0:
            steps = 1
            idx = random.randint(0, len(samples)-1)

            # T_indie
            max_steps = int(attn_mask_input_space[idx, 0, 0, :].sum())

            # (1, C, T)
            x = samples[idx][..., :max_steps:steps].detach().cpu().numpy()
            x_hat = samples_hat[idx][..., :max_steps:steps].detach().cpu().numpy()
            x_hat_masked = (samples_hat[idx] * combined_mask[idx])[..., :max_steps:steps].detach().cpu().numpy()

            ncc_0 = statistics.ncc(samples[idx, 0, 0], samples_hat[idx, 0, 0])
            ncc_0_maskedOnly = statistics.ncc(samples[idx, 0, 0], samples_hat[idx, 0, 0], combined_mask[idx, 0, 0])

            mask_0 = (mask_input_space[idx, 0, 0, :max_steps:steps]==1).cpu().numpy()

            # samples of shape (Batch, 1, Channel, Time)
            max_channels = int(attn_mask_input_space[idx, 0, :, 0].sum())
            if max_channels > 1:
                ch_idx = random.randint(1, max_channels-1)
                ncc_1 = statistics.ncc(samples[idx, 0, ch_idx], samples_hat[idx, 0, ch_idx])
                ncc_1_maskedOnly = statistics.ncc(samples[idx, 0, ch_idx], samples_hat[idx, 0, ch_idx], combined_mask[idx, 0, ch_idx])
                mask_1 = (mask_input_space[idx, 0, ch_idx, :max_steps:steps]==1).cpu().numpy()
            else:
                ch_idx = 0
                ncc_1 = ncc_0
                ncc_1_maskedOnly = ncc_0_maskedOnly
                mask_1 = mask_0

            # Plot reconstructed time series
            plt.close('all')
            plt.figure(figsize=(8, 8))

            plt.subplot(811)
            plt.title(f"Input ({domain[idx]}, channel {0})")
            plt.plot(range(0, x.shape[-1], 1), x[0, 0, :], color='black')

            plt.subplot(812)
            plt.title(f"Input vs Reconstruction (NCC {ncc_0.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x[0, 0, :], color='black')
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=min(x[0, 0, :].min(), x_hat[0, 0, :].min()), 
                             y2=max(x[0, 0, :].max(), x_hat[0, 0, :].max()), 
                             where=mask_0, color='gray', alpha=0.25)
            
            plt.subplot(813)
            plt.title(f"Reconstruction (NCC {ncc_0.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat[0, 0, :].min(), 
                             y2=x_hat[0, 0, :].max(), 
                             where=mask_0, color='gray', alpha=0.25)

            plt.subplot(814)
            plt.title(f"Reconstruction of masked patches (NCC {ncc_0_maskedOnly.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat_masked[0, 0, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat_masked[0, 0, :].min(), 
                             y2=x_hat_masked[0, 0, :].max(), 
                             where=mask_0, color='gray', alpha=0.25)
            
            indices_visible_patches = np.where(mask_0 == False)
            plt.scatter(indices_visible_patches, 
                        x_hat_masked[0, 0, :][indices_visible_patches], 
                        color='white', s=7, zorder=2)

            plt.subplot(815)
            plt.title(f"Input ({domain[idx]}, channel {ch_idx})")
            plt.plot(range(0, x.shape[-1], 1), x[0, ch_idx, :], color='black')

            plt.subplot(816)
            plt.title(f"Input vs Reconstruction (NCC {ncc_1.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x[0, ch_idx, :], color='black')
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=min(x[0, ch_idx, :].min(), x_hat[0, ch_idx, :].min()), 
                             y2=max(x[0, ch_idx, :].max(), x_hat[0, ch_idx, :].max()), 
                             where=mask_1, color='gray', alpha=0.25)

            plt.subplot(817)
            plt.title(f"Reconstruction (NCC {ncc_1.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat[0, ch_idx, :].min(), 
                             y2=x_hat[0, ch_idx, :].max(), 
                             where=mask_1, color='gray', alpha=0.25)

            plt.subplot(818)
            plt.title(f"Reconstruction of masked patches (NCC {ncc_1_maskedOnly.item():.2f}, masked patches in gray)")
            plt.plot(range(0, x.shape[-1], 1), x_hat_masked[0, ch_idx, :], color='darkorange')
            plt.fill_between(range(0, x.shape[-1], 1), 
                             y1=x_hat_masked[0, ch_idx, :].min(), 
                             y2=x_hat_masked[0, ch_idx, :].max(), 
                             where=mask_1, color='gray', alpha=0.25)
            
            indices_visible_patches = np.where(mask_1 == False)
            plt.scatter(indices_visible_patches, 
                        x_hat_masked[0, ch_idx, :][indices_visible_patches], 
                        color='white', s=7, zorder=2)

            plt.tight_layout()
            test_history["Val Reconstruction"] = wandb.Image(plt)

    return test_stats, test_history