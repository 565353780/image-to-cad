#!/usr/bin/env python
# -*- coding: utf-8 -*-

from image_to_cad.Loss.loss_functions import masked_l1_loss

def mask_metrics(mask_pred, mask_gt, pref='mask/'):
    not_mask_pred = ~mask_pred
    not_mask_gt = ~mask_gt

    tp = (mask_pred & mask_gt).sum().float()
    tn = (not_mask_pred & not_mask_gt).sum().float()
    fp = (mask_pred & not_mask_gt).sum().float()
    fn = (not_mask_pred & mask_gt).sum().float()

    new_log = {
        'accuracy': (mask_pred == mask_gt).float().mean(),
        'true_positive': tp / (tp + fn).clamp(1e-5),
        'true_negative': tn / (tn + fp).clamp(1e-5),
        'false_positive': fp / (fp + tn).clamp(1e-5),
        'false_negative': fn / (fn + tp).clamp(1e-5)
    }
    new_log = {(pref + k): v.item() for k, v in new_log.items()}
    return new_log

def depth_metrics(depth_pred, depth_gt, mask,
                  mask_inputs=True, pref='depth/'):
    metrics = {}
    depth_pred = depth_pred.detach()
    metrics['depth_error'] = masked_l1_loss(
        depth_pred,
        depth_gt,
        mask,
        mask_inputs=mask_inputs,
        instance_average=True
    )
    metrics = {(pref + k): v.item() for k, v in metrics.items()}
    return metrics

