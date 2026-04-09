import argparse
import logging
import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tscan import (
    TSCANNet,
    DomainDiscriminator,
    append_jsonl,
    build_loaders,
    copy_student_to_teacher,
    ema_update,
    evaluate_model,
    grad_reverse,
    initialize_metric_head_from_centroids,
    load_backbone_checkpoint,
    load_metric_head_checkpoint,
    remap_metric_head_weights_by_class_names,
    refresh_pseudo_labels,
    save_pseudo_snapshot,
    save_verification_plots,
)


def recursive_namespace(data):
    if isinstance(data, dict):
        return argparse.Namespace(**{k: recursive_namespace(v) for k, v in data.items()})
    if isinstance(data, list):
        return [recursive_namespace(v) for v in data]
    return data


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to TSCAN config yaml')
    return parser.parse_args()


def setup_save_dirs(save_path):
    os.makedirs(save_path, exist_ok=True)
    model_dir = os.path.join(save_path, 'model')
    plots_dir = os.path.join(save_path, 'plots')
    pseudo_dir = os.path.join(save_path, 'pseudo_labels')
    tb_dir = os.path.join(save_path, 'tensorboard')
    for path in [model_dir, plots_dir, pseudo_dir, tb_dir]:
        os.makedirs(path, exist_ok=True)
    return model_dir, plots_dir, pseudo_dir, tb_dir


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


def main():
    args = build_args()
    opts = recursive_namespace(yaml.safe_load(open(args.config, 'r')))
    model_dir, plots_dir, pseudo_dir, tb_dir = setup_save_dirs(opts.save_path)
    safe_copy_config(args.config, opts.save_path)

    logging.basicConfig(
        filename=os.path.join(opts.save_path, 'training.log'),
        level=logging.INFO,
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    writer = SummaryWriter(log_dir=tb_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = build_loaders(
        opts.source_path,
        opts.target_path,
        opts.batch_size,
        opts.num_workers,
        use_augmentation=bool(getattr(opts, 'use_augmentation', True)),
    )
    num_classes = loaders['source_num_classes']

    logging.info(f'Source train classes: {num_classes}')
    logging.info(f'Target train size: {loaders["target_train_size"]}')

    student = TSCANNet(num_classes, scale=opts.metric_scale, margin=opts.metric_margin, gamma=opts.metric_gamma).to(device)
    teacher = TSCANNet(num_classes, scale=opts.metric_scale, margin=opts.metric_margin, gamma=opts.metric_gamma).to(device)
    discriminator = DomainDiscriminator(in_features=512, hidden=opts.discriminator_hidden).to(device)

    load_backbone_checkpoint(student, opts.teacher_init_ckpt, device)
    loaded_metric_head, _, _ = load_metric_head_checkpoint(student, opts.teacher_init_ckpt, device)
    if not loaded_metric_head:
        initialize_metric_head_from_centroids(student, loaders['source_proto'], device)
    else:
        remap_metric_head_weights_by_class_names(
            student,
            opts.source_path,
            loaders['source_class_names'],
            class_limit=num_classes,
        )
    copy_student_to_teacher(student, teacher)

    optimizer = optim.Adam(
        list(student.parameters()) + list(discriminator.parameters()),
        lr=float(opts.lr),
        weight_decay=float(getattr(opts, 'weight_decay', 0.0)),
    )
    scheduler = make_scheduler(optimizer, opts)
    domain_criterion = nn.BCEWithLogitsLoss().to(device)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_teacher_target_eer = None
    best_teacher_target_acc = None
    best_student_target_eer = None
    best_student_target_acc = None
    metrics_path = os.path.join(opts.save_path, 'metrics.jsonl')

    for epoch in tqdm(range(int(opts.epochs)), desc='TSCAN'):
        tau_progress = min(1.0, epoch / max(1, int(getattr(opts, 'tau_ramp_epochs', 1))))
        tau = max(float(opts.tau_end), float(opts.tau_start) - (float(opts.tau_start) - float(opts.tau_end)) * tau_progress)
        use_pseudo_labels = bool(getattr(opts, 'use_pseudo_labels', True))
        use_domain_adversarial = bool(getattr(opts, 'use_domain_adversarial', True))

        if use_domain_adversarial:
            domain_weight = float(opts.lambda_d) * min(1.0, (epoch + 1) / max(1, int(getattr(opts, 'domain_ramp_epochs', 1))))
        else:
            domain_weight = 0.0

        if (not use_pseudo_labels) or epoch < int(getattr(opts, 'target_warmup_epochs', 0)):
            target_weight = 0.0
        else:
            target_progress = min(1.0, (epoch + 1 - int(getattr(opts, 'target_warmup_epochs', 0))) / max(1, int(getattr(opts, 'target_ramp_epochs', 1))))
            target_weight = float(opts.lambda_t) * target_progress

        if use_pseudo_labels:
            pseudo_bank, pseudo_stats = refresh_pseudo_labels(teacher, loaders['target_refresh'], device, tau=tau)
        else:
            pseudo_bank = {}
            pseudo_stats = {
                'valid_count': 0,
                'total_count': 0,
                'valid_ratio': 0.0,
                'mean_confidence': 0.0,
            }
        if use_pseudo_labels and epoch % int(getattr(opts, 'save_pseudo_every', 5)) == 0:
            save_pseudo_snapshot(pseudo_bank, os.path.join(pseudo_dir, f'epoch_{epoch:03d}.csv'))

        student.train()
        discriminator.train()
        source_iter = iter(loaders['source_train'])
        target_iter = iter(loaders['target_train'])
        steps = min(len(loaders['source_train']), len(loaders['target_train']))

        epoch_source_loss = 0.0
        epoch_target_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_total_loss = 0.0
        epoch_valid = 0
        epoch_target_samples = 0

        for _ in range(steps):
            try:
                source_imgs, source_labels, _, _ = next(source_iter)
            except StopIteration:
                source_iter = iter(loaders['source_train'])
                source_imgs, source_labels, _, _ = next(source_iter)
            try:
                _, target_strong, _, target_indices, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(loaders['target_train'])
                _, target_strong, _, target_indices, _ = next(target_iter)

            source_imgs = source_imgs.to(device)
            source_labels = source_labels.to(device)
            target_strong = target_strong.to(device)
            target_indices_list = target_indices.tolist()

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                src_pooled, src_emb, source_loss = student.compute_source_loss(source_imgs, source_labels)
                tgt_pooled, tgt_emb = student(target_strong)

                valid_positions = []
                pseudo_labels = []
                for pos, idx in enumerate(target_indices_list):
                    row = pseudo_bank.get(int(idx))
                    if row is not None and row['valid']:
                        valid_positions.append(pos)
                        pseudo_labels.append(row['pseudo_label'])
                epoch_target_samples += len(target_indices_list)
                epoch_valid += len(valid_positions)

                if use_pseudo_labels and valid_positions:
                    valid_positions = torch.tensor(valid_positions, device=device, dtype=torch.long)
                    pseudo_labels_tensor = torch.tensor(pseudo_labels, device=device, dtype=torch.long)
                    target_loss = student.metric_head(tgt_emb.index_select(0, valid_positions), pseudo_labels_tensor)
                else:
                    target_loss = torch.zeros((), device=device)

                if use_domain_adversarial:
                    grl_lambda = float(getattr(opts, 'grl_lambda', 1.0))
                    domain_features = torch.cat([
                        grad_reverse(src_pooled, lambd=grl_lambda),
                        grad_reverse(tgt_pooled, lambd=grl_lambda),
                    ], dim=0)
                    domain_logits = discriminator(domain_features).view(-1)
                    domain_labels = torch.cat([
                        torch.zeros(src_pooled.size(0), device=device),
                        torch.ones(tgt_pooled.size(0), device=device),
                    ], dim=0)
                    domain_loss = domain_criterion(domain_logits, domain_labels)
                else:
                    domain_loss = torch.zeros((), device=device)

                total_loss = float(opts.lambda_s) * source_loss + target_weight * target_loss + domain_weight * domain_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if bool(getattr(opts, 'use_ema', True)):
                ema_update(teacher, student, decay=float(opts.ema_decay))

            epoch_source_loss += float(source_loss.item())
            epoch_target_loss += float(target_loss.item())
            epoch_domain_loss += float(domain_loss.item())
            epoch_total_loss += float(total_loss.item())

        if scheduler is not None:
            scheduler.step()

        avg_source_loss = epoch_source_loss / max(1, steps)
        avg_target_loss = epoch_target_loss / max(1, steps)
        avg_domain_loss = epoch_domain_loss / max(1, steps)
        avg_total_loss = epoch_total_loss / max(1, steps)
        valid_ratio = epoch_valid / max(1, epoch_target_samples)

        writer.add_scalar('train/source_loss', avg_source_loss, epoch)
        writer.add_scalar('train/target_loss', avg_target_loss, epoch)
        writer.add_scalar('train/domain_loss', avg_domain_loss, epoch)
        writer.add_scalar('train/total_loss', avg_total_loss, epoch)
        writer.add_scalar('train/pseudo_valid_ratio', valid_ratio, epoch)
        writer.add_scalar('train/pseudo_mean_confidence', pseudo_stats['mean_confidence'], epoch)
        writer.add_scalar('train/tau', tau, epoch)
        writer.add_scalar('train/lambda_t', target_weight, epoch)
        writer.add_scalar('train/lambda_d', domain_weight, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        teacher_source_metrics, teacher_source_feats, teacher_source_labels = evaluate_model(teacher, loaders['source_test'], device)
        teacher_target_metrics, teacher_target_feats, teacher_target_labels = evaluate_model(teacher, loaders['target_test'], device)
        student_source_metrics, student_source_feats, student_source_labels = evaluate_model(student, loaders['source_test'], device)
        student_target_metrics, student_target_feats, student_target_labels = evaluate_model(student, loaders['target_test'], device)

        log_to_tensorboard(writer, 'teacher/source', teacher_source_metrics, epoch)
        log_to_tensorboard(writer, 'teacher/target', teacher_target_metrics, epoch)
        log_to_tensorboard(writer, 'student/source', student_source_metrics, epoch)
        log_to_tensorboard(writer, 'student/target', student_target_metrics, epoch)

        teacher_target_summary = metric_summary(teacher_target_metrics)
        student_target_summary = metric_summary(student_target_metrics)
        record = {
            'epoch': epoch,
            'lr': round(float(optimizer.param_groups[0]['lr']), 8),
            'train': {
                'source_loss': round(avg_source_loss, 4),
                'target_loss': round(avg_target_loss, 4),
                'domain_loss': round(avg_domain_loss, 4),
                'total_loss': round(avg_total_loss, 4),
                'pseudo_valid_ratio': round(valid_ratio, 4),
                'pseudo_mean_confidence': round(float(pseudo_stats['mean_confidence']), 4),
                'tau': round(float(tau), 4),
                'lambda_t': round(float(target_weight), 4),
                'lambda_d': round(float(domain_weight), 4),
            },
            'teacher': {
                'source': metric_summary(teacher_source_metrics),
                'target': teacher_target_summary,
            },
            'student': {
                'source': metric_summary(student_source_metrics),
                'target': student_target_summary,
            },
        }
        append_jsonl(record, metrics_path)
        logging.info(str(record))
        print(record)
        logging.info(log_split_metrics('teacher', 'source', teacher_source_metrics))
        logging.info(log_split_metrics('teacher', 'target', teacher_target_metrics))
        logging.info(log_split_metrics('student', 'source', student_source_metrics))
        logging.info(log_split_metrics('student', 'target', student_target_metrics))
        print(log_split_metrics('teacher', 'source', teacher_source_metrics))
        print(log_split_metrics('teacher', 'target', teacher_target_metrics))
        print(log_split_metrics('student', 'source', student_source_metrics))
        print(log_split_metrics('student', 'target', student_target_metrics))

        save_checkpoint(os.path.join(model_dir, 'teacher_ema_latest.pth'), epoch, teacher, teacher_target_summary)
        save_checkpoint(os.path.join(model_dir, 'student_latest.pth'), epoch, student, student_target_summary)

        if is_better_eer(teacher_target_metrics, best_teacher_target_eer):
            best_teacher_target_eer = {
                'eer': float(teacher_target_metrics['eer']),
                'acc': float(teacher_target_metrics['acc']),
                'epoch': epoch,
            }
            save_checkpoint(os.path.join(model_dir, 'teacher_ema_best_eer.pth'), epoch, teacher, teacher_target_summary)
            save_verification_plots(teacher_source_feats, teacher_source_labels, plots_dir, 'teacher_source_best_eer')
            save_verification_plots(teacher_target_feats, teacher_target_labels, plots_dir, 'teacher_target_best_eer')

        if is_better_acc(teacher_target_metrics, best_teacher_target_acc):
            best_teacher_target_acc = {
                'eer': float(teacher_target_metrics['eer']),
                'acc': float(teacher_target_metrics['acc']),
                'epoch': epoch,
            }
            save_checkpoint(os.path.join(model_dir, 'teacher_ema_best_acc.pth'), epoch, teacher, teacher_target_summary)

        if is_better_eer(student_target_metrics, best_student_target_eer):
            best_student_target_eer = {
                'eer': float(student_target_metrics['eer']),
                'acc': float(student_target_metrics['acc']),
                'epoch': epoch,
            }
            save_checkpoint(os.path.join(model_dir, 'student_best_eer.pth'), epoch, student, student_target_summary)
            save_verification_plots(student_source_feats, student_source_labels, plots_dir, 'student_source_best_eer')
            save_verification_plots(student_target_feats, student_target_labels, plots_dir, 'student_target_best_eer')

        if is_better_acc(student_target_metrics, best_student_target_acc):
            best_student_target_acc = {
                'eer': float(student_target_metrics['eer']),
                'acc': float(student_target_metrics['acc']),
                'epoch': epoch,
            }
            save_checkpoint(os.path.join(model_dir, 'student_best_acc.pth'), epoch, student, student_target_summary)

    writer.close()


if __name__ == '__main__':
    main()
