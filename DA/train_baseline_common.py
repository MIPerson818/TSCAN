import argparse
import os
import shutil

import torch
import torch.optim as optim
import yaml

from tscan import (
    initialize_metric_head_from_centroids,
    load_backbone_checkpoint,
    load_metric_head_checkpoint,
    remap_metric_head_weights_by_class_names,
)


def recursive_namespace(data):
    if isinstance(data, dict):
        return argparse.Namespace(**{k: recursive_namespace(v) for k, v in data.items()})
    if isinstance(data, list):
        return [recursive_namespace(v) for v in data]
    return data


def build_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    return parser.parse_args()


def load_yaml_config(config_path):
    return recursive_namespace(yaml.safe_load(open(config_path, 'r')))


def setup_save_dirs(save_path):
    os.makedirs(save_path, exist_ok=True)
    model_dir = os.path.join(save_path, 'model')
    plots_dir = os.path.join(save_path, 'plots')
    tb_dir = os.path.join(save_path, 'tensorboard')
    for path in [model_dir, plots_dir, tb_dir]:
        os.makedirs(path, exist_ok=True)
    return model_dir, plots_dir, tb_dir


def safe_copy_config(config_path, save_path):
    dst = os.path.join(save_path, os.path.basename(config_path))
    if os.path.abspath(config_path) != os.path.abspath(dst):
        shutil.copy(config_path, dst)


def make_scheduler(optimizer, cfg):
    name = getattr(cfg, 'scheduler', 'multistep')
    if name == 'multistep':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(getattr(cfg, 'scheduler_milestones', [40, 80, 120])),
            gamma=float(getattr(cfg, 'scheduler_gamma', 0.2)),
        )
    if name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(getattr(cfg, 'scheduler_t0', 10)),
            T_mult=int(getattr(cfg, 'scheduler_tmult', 2)),
            eta_min=float(getattr(cfg, 'scheduler_eta_min', 1e-6)),
        )
    return None


def metric_summary(metrics):
    return {
        'eer': round(float(metrics['eer']), 4),
        'acc': round(float(metrics['acc']), 4),
        'true_acc': round(float(metrics['true_acc']), 4),
        'mAP': round(float(metrics['mAP']), 4),
        'tar@0.1': round(float(metrics['tar_at_far'].get(0.1, 0.0)), 4),
        'tar@0.01': round(float(metrics['tar_at_far'].get(0.01, 0.0)), 4),
        'tar@0.001': round(float(metrics['tar_at_far'].get(0.001, 0.0)), 4),
    }


def is_better_eer(metrics, best_metrics):
    if best_metrics is None:
        return True
    if metrics['eer'] < best_metrics['eer'] - 1e-8:
        return True
    if abs(metrics['eer'] - best_metrics['eer']) <= 1e-8 and metrics['acc'] > best_metrics['acc'] + 1e-8:
        return True
    return False


def is_better_acc(metrics, best_metrics):
    if best_metrics is None:
        return True
    if metrics['acc'] > best_metrics['acc'] + 1e-8:
        return True
    if abs(metrics['acc'] - best_metrics['acc']) <= 1e-8 and metrics['eer'] < best_metrics['eer'] - 1e-8:
        return True
    return False


def save_checkpoint(path, epoch, model, metrics, extra=None):
    payload = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'metrics': metrics,
    }
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


def log_to_tensorboard(writer, prefix, metrics, epoch):
    writer.add_scalar(f'{prefix}/eer', metrics['eer'], epoch)
    writer.add_scalar(f'{prefix}/acc', metrics['acc'], epoch)
    writer.add_scalar(f'{prefix}/true_acc', metrics['true_acc'], epoch)
    writer.add_scalar(f'{prefix}/mAP', metrics['mAP'], epoch)
    writer.add_scalar(f'{prefix}/tar_far_0.1', metrics['tar_at_far'].get(0.1, 0.0), epoch)
    writer.add_scalar(f'{prefix}/tar_far_0.01', metrics['tar_at_far'].get(0.01, 0.0), epoch)
    writer.add_scalar(f'{prefix}/tar_far_0.001', metrics['tar_at_far'].get(0.001, 0.0), epoch)


def log_split_metrics(role, split, metrics):
    summary = metric_summary(metrics)
    return (
        f'{role}/{split}: '
        f"EER={summary['eer']:.4f}, "
        f"ACC={summary['acc']:.4f}, "
        f"TrueACC={summary['true_acc']:.4f}, "
        f"mAP={summary['mAP']:.4f}, "
        f"TAR@0.1={summary['tar@0.1']:.4f}, "
        f"TAR@0.01={summary['tar@0.01']:.4f}, "
        f"TAR@0.001={summary['tar@0.001']:.4f}"
    )


def initialize_model_from_checkpoint(model, loaders, opts, device):
    load_backbone_checkpoint(model, opts.teacher_init_ckpt, device)
    loaded_metric_head, _, _ = load_metric_head_checkpoint(model, opts.teacher_init_ckpt, device)
    if not loaded_metric_head:
        initialize_metric_head_from_centroids(model, loaders['source_proto'], device)
    else:
        remap_metric_head_weights_by_class_names(
            model,
            opts.source_path,
            loaders['source_class_names'],
            class_limit=loaders['source_num_classes'],
        )
