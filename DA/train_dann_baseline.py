import logging
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tscan import (
    DomainDiscriminator,
    TSCANNet,
    append_jsonl,
    build_loaders,
    evaluate_model,
    grad_reverse,
    save_verification_plots,
)
from train_baseline_common import (
    build_args,
    initialize_model_from_checkpoint,
    is_better_acc,
    is_better_eer,
    load_yaml_config,
    log_split_metrics,
    log_to_tensorboard,
    make_scheduler,
    metric_summary,
    safe_copy_config,
    save_checkpoint,
    setup_save_dirs,
)


def main():
    args = build_args('Train DANN baseline')
    opts = load_yaml_config(args.config)
    model_dir, plots_dir, tb_dir = setup_save_dirs(opts.save_path)
    safe_copy_config(args.config, opts.save_path)

    logging.basicConfig(
        filename=os.path.join(opts.save_path, 'training.log'),
        level=logging.INFO,
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    writer = SummaryWriter(log_dir=tb_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = build_loaders(
        opts.source_path,
        opts.target_path,
        int(opts.batch_size),
        int(opts.num_workers),
        use_augmentation=bool(getattr(opts, 'use_augmentation', True)),
    )
    num_classes = loaders['source_num_classes']

    model = TSCANNet(
        num_classes,
        scale=float(opts.metric_scale),
        margin=float(opts.metric_margin),
        gamma=float(opts.metric_gamma),
    ).to(device)
    initialize_model_from_checkpoint(model, loaders, opts, device)
    discriminator = DomainDiscriminator(
        in_features=512,
        hidden=int(getattr(opts, 'discriminator_hidden', 256)),
    ).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=float(opts.lr),
        weight_decay=float(getattr(opts, 'weight_decay', 0.0)),
    )
    scheduler = make_scheduler(optimizer, opts)
    criterion = nn.BCEWithLogitsLoss().to(device)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_target_eer = None
    best_target_acc = None
    metrics_path = os.path.join(opts.save_path, 'metrics.jsonl')
    early_stop_patience = int(getattr(opts, 'early_stop_patience', 0))
    early_stop_min_delta = float(getattr(opts, 'early_stop_min_delta', 0.0))
    no_improve_epochs = 0

    for epoch in tqdm(range(int(opts.epochs)), desc='DANN'):
        model.train()
        discriminator.train()
        source_iter = iter(loaders['source_train'])
        target_iter = iter(loaders['target_train'])
        steps = min(len(loaders['source_train']), len(loaders['target_train']))
        domain_weight = float(getattr(opts, 'lambda_d', 0.1)) * min(
            1.0,
            (epoch + 1) / max(1, int(getattr(opts, 'domain_ramp_epochs', 1))),
        )

        epoch_source_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_total_loss = 0.0

        for _ in range(steps):
            try:
                source_imgs, source_labels, _, _ = next(source_iter)
            except StopIteration:
                source_iter = iter(loaders['source_train'])
                source_imgs, source_labels, _, _ = next(source_iter)
            try:
                _, target_strong, _, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(loaders['target_train'])
                _, target_strong, _, _, _ = next(target_iter)

            source_imgs = source_imgs.to(device)
            source_labels = source_labels.to(device)
            target_strong = target_strong.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                src_pooled, _, source_loss = model.compute_source_loss(source_imgs, source_labels)
                tgt_pooled, _ = model(target_strong)

                domain_features = torch.cat([
                    grad_reverse(src_pooled, lambd=float(getattr(opts, 'grl_lambda', 1.0))),
                    grad_reverse(tgt_pooled, lambd=float(getattr(opts, 'grl_lambda', 1.0))),
                ], dim=0)
                domain_logits = discriminator(domain_features).view(-1)
                domain_labels = torch.cat([
                    torch.zeros(src_pooled.size(0), device=device),
                    torch.ones(tgt_pooled.size(0), device=device),
                ], dim=0)
                domain_loss = criterion(domain_logits, domain_labels)
                total_loss = float(getattr(opts, 'lambda_s', 1.0)) * source_loss + domain_weight * domain_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_source_loss += float(source_loss.item())
            epoch_domain_loss += float(domain_loss.item())
            epoch_total_loss += float(total_loss.item())

        if scheduler is not None:
            scheduler.step()

        avg_source_loss = epoch_source_loss / max(1, steps)
        avg_domain_loss = epoch_domain_loss / max(1, steps)
        avg_total_loss = epoch_total_loss / max(1, steps)

        writer.add_scalar('train/source_loss', avg_source_loss, epoch)
        writer.add_scalar('train/domain_loss', avg_domain_loss, epoch)
        writer.add_scalar('train/total_loss', avg_total_loss, epoch)
        writer.add_scalar('train/lambda_d', domain_weight, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        source_metrics, source_feats, source_labels_np = evaluate_model(model, loaders['source_test'], device)
        target_metrics, target_feats, target_labels_np = evaluate_model(model, loaders['target_test'], device)

        log_to_tensorboard(writer, 'model/source', source_metrics, epoch)
        log_to_tensorboard(writer, 'model/target', target_metrics, epoch)

        target_summary = metric_summary(target_metrics)
        record = {
            'epoch': epoch,
            'lr': round(float(optimizer.param_groups[0]['lr']), 8),
            'train': {
                'source_loss': round(avg_source_loss, 4),
                'domain_loss': round(avg_domain_loss, 4),
                'total_loss': round(avg_total_loss, 4),
                'lambda_d': round(float(domain_weight), 4),
            },
            'source': metric_summary(source_metrics),
            'target': target_summary,
        }
        append_jsonl(record, metrics_path)
        logging.info(str(record))
        logging.info(log_split_metrics('model', 'source', source_metrics))
        logging.info(log_split_metrics('model', 'target', target_metrics))
        print(record)
        print(log_split_metrics('model', 'source', source_metrics))
        print(log_split_metrics('model', 'target', target_metrics))

        save_checkpoint(os.path.join(model_dir, 'latest.pth'), epoch, model, target_summary)

        improved_eer = False
        if best_target_eer is None or target_metrics['eer'] < best_target_eer['eer'] - early_stop_min_delta:
            improved_eer = True
        if is_better_eer(target_metrics, best_target_eer):
            best_target_eer = {
                'eer': float(target_metrics['eer']),
                'acc': float(target_metrics['acc']),
                'epoch': epoch,
            }
            save_checkpoint(os.path.join(model_dir, 'best_eer.pth'), epoch, model, target_summary)
            save_verification_plots(source_feats, source_labels_np, plots_dir, 'source_best_eer')
            save_verification_plots(target_feats, target_labels_np, plots_dir, 'target_best_eer')

        if is_better_acc(target_metrics, best_target_acc):
            best_target_acc = {
                'eer': float(target_metrics['eer']),
                'acc': float(target_metrics['acc']),
                'epoch': epoch,
            }
            save_checkpoint(os.path.join(model_dir, 'best_acc.pth'), epoch, model, target_summary)

        if early_stop_patience > 0:
            if improved_eer:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                logging.info(
                    f'Early stopping triggered at epoch {epoch}: '
                    f'no target EER improvement for {early_stop_patience} epochs.'
                )
                print(
                    f'Early stopping triggered at epoch {epoch}: '
                    f'no target EER improvement for {early_stop_patience} epochs.'
                )
                break

    writer.close()


if __name__ == '__main__':
    main()
