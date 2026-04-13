import argparse
import csv
import os
import re
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from DA.tscan import (
    PalmEvalDataset,
    TSCANNet,
    build_eval_transform,
    compute_pair_scores,
    get_plot_font,
)


def build_args():
    parser = argparse.ArgumentParser(description='Plot ROC comparison figures')
    parser.add_argument('--config', type=str, required=True, help='Path to ROC plot config yaml')
    return parser.parse_args()


def recursive_namespace(data):
    if isinstance(data, dict):
        return {k: recursive_namespace(v) for k, v in data.items()}
    if isinstance(data, list):
        return [recursive_namespace(v) for v in data]
    return data


def sanitize_name(text):
    text = re.sub(r'\s+', '_', text.strip())
    text = re.sub(r'[^0-9A-Za-z_\-\u4e00-\u9fff]+', '', text)
    return text


def extract_features(model, loader, device):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, embedding = model(imgs)
            all_features.append(embedding.detach().cpu())
            all_labels.append(labels)
    return torch.cat(all_features, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def load_model(num_classes, metric_scale, metric_margin, metric_gamma, ckpt_path, device):
    model = TSCANNet(
        num_classes=num_classes,
        scale=metric_scale,
        margin=metric_margin,
        gamma=metric_gamma,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state, strict=True)
    return model


def compute_roc_points(model, loader, device):
    features, labels = extract_features(model, loader, device)
    positive_scores, negative_scores = compute_pair_scores(
        features,
        labels,
        distance_metric='cosine',
        normalize=True,
    )
    y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
    y_score = positive_scores + negative_scores
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def write_roc_csv(save_path, fpr, tpr):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['FAR', 'TAR'])
        for x, y in zip(fpr, tpr):
            writer.writerow([f'{float(x):.10f}', f'{float(y):.10f}'])


def build_loader(target_path, batch_size, num_workers, class_limit):
    dataset = PalmEvalDataset(
        root=target_path,
        transform=build_eval_transform(),
        split='test',
        class_limit=class_limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


def draw_single_protocol(cfg, device):
    os.makedirs(cfg['save_path'], exist_ok=True)
    data_dir = os.path.join(cfg['save_path'], 'data')
    os.makedirs(data_dir, exist_ok=True)

    loader = build_loader(
        cfg['target_path'],
        int(cfg.get('batch_size', 48)),
        int(cfg.get('num_workers', 8)),
        int(cfg.get('class_limit', 200)),
    )

    title_font = get_plot_font(size=16)
    label_font = get_plot_font(size=13)
    legend_font = get_plot_font(size=11)

    plt.figure(figsize=(7, 6))
    for method in cfg['methods']:
        ckpt_path = method['ckpt']
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        model = load_model(
            num_classes=int(cfg.get('num_classes', 150)),
            metric_scale=float(cfg.get('metric_scale', 64)),
            metric_margin=float(cfg.get('metric_margin', 0.5)),
            metric_gamma=float(cfg.get('metric_gamma', 2)),
            ckpt_path=ckpt_path,
            device=device,
        )
        fpr, tpr, roc_auc = compute_roc_points(model, loader, device)
        write_roc_csv(
            os.path.join(
                data_dir,
                f"{sanitize_name(cfg.get('title', 'protocol'))}_{sanitize_name(method['name'])}_roc.csv",
            ),
            fpr,
            tpr,
        )
        plt.plot(
            fpr,
            tpr,
            linewidth=float(method.get('linewidth', 2.0)),
            linestyle=method.get('linestyle', '-'),
            color=method.get('color', None),
            label=f"{method['name']} (AUC={roc_auc:.4f})",
        )

    plt.xlabel('FAR', fontproperties=label_font)
    plt.ylabel('TAR', fontproperties=label_font)
    plt.title(cfg['title'], fontproperties=title_font)
    plt.xlim(cfg.get('xlim', [0.0, 0.1]))
    plt.ylim(cfg.get('ylim', [0.6, 1.0]))
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.legend(loc='lower right', prop=legend_font, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg['save_path'], 'compare_roc.png'), dpi=260)
    plt.close()


def draw_protocol_grid(cfg, device):
    save_path = cfg['save_path']
    data_dir = os.path.join(save_path, 'data')
    os.makedirs(data_dir, exist_ok=True)

    title_font = get_plot_font(size=16)
    subplot_font = get_plot_font(size=13)
    label_font = get_plot_font(size=12)
    legend_font = get_plot_font(size=11)

    protocols = cfg['protocols']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for ax, protocol in zip(axes, protocols):
        loader = build_loader(
            protocol['target_path'],
            int(cfg.get('batch_size', 48)),
            int(cfg.get('num_workers', 8)),
            int(cfg.get('class_limit', 200)),
        )
        for method in protocol['methods']:
            ckpt_path = method['ckpt']
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
            model = load_model(
                num_classes=int(cfg.get('num_classes', 150)),
                metric_scale=float(cfg.get('metric_scale', 64)),
                metric_margin=float(cfg.get('metric_margin', 0.5)),
                metric_gamma=float(cfg.get('metric_gamma', 2)),
                ckpt_path=ckpt_path,
                device=device,
            )
            fpr, tpr, roc_auc = compute_roc_points(model, loader, device)
            write_roc_csv(
                os.path.join(
                    data_dir,
                    f"{sanitize_name(protocol['id'])}_{sanitize_name(method['name'])}_roc.csv",
                ),
                fpr,
                tpr,
            )
            ax.plot(
                fpr,
                tpr,
                linewidth=float(method.get('linewidth', 2.0)),
                linestyle=method.get('linestyle', '-'),
                color=method.get('color', None),
                label=f"{method['name']} (AUC={roc_auc:.4f})",
            )

        ax.set_xlim(protocol.get('xlim', cfg.get('xlim', [0.0, 0.1])))
        ax.set_ylim(protocol.get('ylim', cfg.get('ylim', [0.6, 1.0])))
        ax.set_xlabel('FAR', fontproperties=label_font)
        ax.set_ylabel('TAR', fontproperties=label_font)
        ax.set_title(protocol['title'], fontproperties=subplot_font, pad=4)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        legend_handles, legend_labels = ax.get_legend_handles_labels()

    # handle fewer than 4 protocols
    for ax in axes[len(protocols):]:
        ax.axis('off')

    fig.suptitle(cfg['title'], fontproperties=title_font, y=0.955)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            ncol=min(3, len(legend_labels)),
            frameon=True,
            prop=legend_font,
            bbox_to_anchor=(0.5, 0.028),
        )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12, wspace=0.20, hspace=0.24)
    fig.savefig(os.path.join(save_path, 'compare_roc_grid.png'), dpi=280)
    plt.close(fig)


def main():
    args = build_args()
    cfg = recursive_namespace(yaml.safe_load(open(args.config, 'r')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'protocols' in cfg:
        draw_protocol_grid(cfg, device)
    else:
        draw_single_protocol(cfg, device)


if __name__ == '__main__':
    main()
