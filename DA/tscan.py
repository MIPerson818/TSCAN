import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import font_manager
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader

from configs import settings
from losses.arcface import ArcFaceLoss
from models.resnet import ResNet_double, BasicBlock
from utils.grad_reverse import grad_reverse
from utils.utils import list_pictures
from val_base import compute_metrics


def get_plot_font(size=None):
    candidate_paths = [
        Path(__file__).resolve().parent.parent / 'tmp' / 'fonts' / 'NotoSansSC-Regular.otf',
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'),
    ]
    for font_path in candidate_paths:
        if font_path.exists():
            return font_manager.FontProperties(fname=str(font_path), size=size)
    return font_manager.FontProperties(size=size)


class TSCANNet(nn.Module):
    def __init__(self, num_classes, scale=64, margin=0.50, gamma=2):
        super().__init__()
        self.backbone = ResNet_double(BasicBlock, [2, 2, 2, 2])
        self.metric_head = ArcFaceLoss(
            embedding_size=128,
            class_num=num_classes,
            s=scale,
            m=margin,
            gamma=gamma,
        )

    def forward(self, x):
        return self.backbone(x)

    def compute_source_loss(self, x, labels):
        pooled_feature, embedding = self.backbone(x)
        loss = self.metric_head(embedding, labels)
        return pooled_feature, embedding, loss

    def get_prototypes(self):
        return F.normalize(self.metric_head.weight, p=2, dim=1, eps=1e-12)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_features=512, hidden=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.classifier(x)


class PalmSourceDataset(Dataset):
    def __init__(self, root, transform, split='train', class_limit=200):
        self.root = root
        self.transform = transform
        split_path = os.path.join(root, f'cent_{split}')
        self.paths, self.labels, self.class_names = collect_split_samples(split_path, class_limit)
        self.loader = default_loader

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.loader(self.paths[index])
        return self.transform(img), self.labels[index], index, self.paths[index]


class PalmTargetDataset(Dataset):
    def __init__(self, root, weak_transform, strong_transform, split='train', class_limit=200):
        self.root = root
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        split_path = os.path.join(root, f'cent_{split}')
        self.paths, self.labels, self.class_names = collect_split_samples(split_path, class_limit)
        self.loader = default_loader

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.loader(self.paths[index])
        weak_img = self.weak_transform(img.copy())
        strong_img = self.strong_transform(img.copy())
        return weak_img, strong_img, self.labels[index], index, self.paths[index]


class PalmEvalDataset(Dataset):
    def __init__(self, root, transform, split='test', class_limit=200):
        self.root = root
        self.transform = transform
        split_path = os.path.join(root, f'cent_{split}')
        self.paths, self.labels, self.class_names = collect_split_samples(split_path, class_limit)
        self.loader = default_loader

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.loader(self.paths[index])
        return self.transform(img), self.labels[index]


def collect_split_samples(split_path, class_limit=200):
    split_dir = Path(split_path)
    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if class_limit is not None:
        class_dirs = class_dirs[:class_limit]

    paths = []
    labels = []
    class_names = [class_dir.name for class_dir in class_dirs]
    valid_suffixes = {'.jpg', '.bmp', '.png', '.jpeg', '.rgb', '.tif', '.JPG'}

    for label, class_dir in enumerate(class_dirs):
        image_files = sorted(
            [p for p in class_dir.iterdir() if p.is_file() and p.suffix in valid_suffixes],
            key=lambda p: p.name,
        )
        for image_path in image_files:
            paths.append(str(image_path))
            labels.append(label)
    return paths, labels, class_names


def collect_legacy_class_names(split_path, class_limit=200):
    img_paths = list_pictures(split_path)
    class_names = []
    for image_path in img_paths:
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name not in class_names:
            if len(class_names) >= class_limit:
                break
            class_names.append(class_name)
    return class_names


def build_eval_transform(mean=None, std=None):
    mean = settings.DATA_TRAIN_MEAN if mean is None else mean
    std = settings.DATA_TRAIN_STD if std is None else std
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_light_train_transform(mean=None, std=None):
    mean = settings.DATA_TRAIN_MEAN if mean is None else mean
    std = settings.DATA_TRAIN_STD if std is None else std
    return transforms.Compose([
        transforms.Resize((124, 124)),
        transforms.RandomCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_strong_train_transform(mean=None, std=None):
    mean = settings.DATA_TRAIN_MEAN if mean is None else mean
    std = settings.DATA_TRAIN_STD if std is None else std
    return transforms.Compose([
        transforms.Resize((124, 124)),
        transforms.RandomCrop(112),
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.08,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_loaders(source_root, target_root, batch_size, num_workers, use_augmentation=True):
    # Ablation semantics:
    # use_augmentation = false means disable weak/strong distinction and use the weak pipeline for both.
    weak_transform = build_light_train_transform()
    strong_transform = build_strong_train_transform() if use_augmentation else build_light_train_transform()

    source_train = PalmSourceDataset(source_root, strong_transform, split='train')
    source_proto = PalmSourceDataset(source_root, build_eval_transform(), split='train')
    source_test = PalmEvalDataset(source_root, build_eval_transform(), split='test')
    target_train = PalmTargetDataset(target_root, weak_transform, strong_transform, split='train')
    target_test = PalmEvalDataset(target_root, build_eval_transform(), split='test')

    source_train_loader = DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    source_proto_loader = DataLoader(source_proto, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    source_test_loader = DataLoader(source_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    target_train_loader = DataLoader(target_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    target_refresh_loader = DataLoader(target_train, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    target_test_loader = DataLoader(target_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    num_classes = max(source_train.labels) + 1
    return {
        'source_train': source_train_loader,
        'source_proto': source_proto_loader,
        'source_test': source_test_loader,
        'target_train': target_train_loader,
        'target_refresh': target_refresh_loader,
        'target_test': target_test_loader,
        'source_num_classes': num_classes,
        'source_class_names': source_proto.class_names,
        'target_train_size': len(target_train),
    }


def load_backbone_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'backbone_state' in state:
        state = state['backbone_state']
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    if isinstance(state, dict) and any(k.startswith('backbone.') for k in state.keys()):
        state = {k.replace('backbone.', '', 1): v for k, v in state.items() if k.startswith('backbone.')}
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    return missing, unexpected


def load_metric_head_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'metric_head_state' in state:
        missing, unexpected = model.metric_head.load_state_dict(state['metric_head_state'], strict=False)
        return True, missing, unexpected
    if isinstance(state, dict) and 'metric_head' in state:
        missing, unexpected = model.metric_head.load_state_dict(state['metric_head'], strict=False)
        return True, missing, unexpected
    return False, [], []


def remap_metric_head_weights_by_class_names(model, source_root, sorted_class_names, class_limit=200):
    split_path = os.path.join(source_root, 'cent_train')
    legacy_class_names = collect_legacy_class_names(split_path, class_limit=class_limit)
    if len(legacy_class_names) != len(sorted_class_names):
        return False
    legacy_index = {name: idx for idx, name in enumerate(legacy_class_names)}
    if any(name not in legacy_index for name in sorted_class_names):
        return False

    old_weight = model.metric_head.weight.data.clone()
    new_weight = old_weight.clone()
    for sorted_idx, class_name in enumerate(sorted_class_names):
        new_weight[sorted_idx] = old_weight[legacy_index[class_name]]
    model.metric_head.weight.data.copy_(new_weight)
    return True


def initialize_metric_head_from_centroids(model, source_loader, device):
    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels, _, _ in source_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            _, embedding = model(imgs)
            embedding = F.normalize(embedding, p=2, dim=1, eps=1e-12)
            all_feats.append(embedding)
            all_labels.append(labels)
    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    weight = torch.zeros_like(model.metric_head.weight.data)
    for cls in labels.unique(sorted=True):
        cls_mask = labels == cls
        centroid = feats[cls_mask].mean(dim=0)
        weight[cls] = F.normalize(centroid, p=2, dim=0, eps=1e-12)
    model.metric_head.weight.data.copy_(weight)
    return weight


def copy_student_to_teacher(student, teacher):
    teacher.load_state_dict(student.state_dict(), strict=True)


def ema_update(teacher, student, decay=0.99):
    with torch.no_grad():
        teacher_params = dict(teacher.named_parameters())
        student_params = dict(student.named_parameters())
        for name, teacher_param in teacher_params.items():
            student_param = student_params[name]
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
        teacher_buffers = dict(teacher.named_buffers())
        student_buffers = dict(student.named_buffers())
        for name, teacher_buffer in teacher_buffers.items():
            student_buffer = student_buffers[name]
            teacher_buffer.data.copy_(student_buffer.data)


def refresh_pseudo_labels(teacher, target_refresh_loader, device, tau=0.8):
    teacher.eval()
    prototypes = teacher.get_prototypes().detach()
    pseudo_bank = {}
    confidences = []
    valid_count = 0
    with torch.no_grad():
        for weak_imgs, _, _, indices, paths in target_refresh_loader:
            weak_imgs = weak_imgs.to(device)
            _, embedding = teacher(weak_imgs)
            embedding = F.normalize(embedding, p=2, dim=1, eps=1e-12)
            similarity = torch.matmul(embedding, prototypes.t())
            confidence, pseudo_label = torch.max(similarity, dim=1)
            confidence_cpu = confidence.detach().cpu().tolist()
            pseudo_cpu = pseudo_label.detach().cpu().tolist()
            indices_cpu = indices.tolist()
            for idx, conf, pseudo, path in zip(indices_cpu, confidence_cpu, pseudo_cpu, paths):
                valid = conf >= tau
                if valid:
                    valid_count += 1
                confidences.append(conf)
                pseudo_bank[int(idx)] = {
                    'pseudo_label': int(pseudo),
                    'confidence': float(conf),
                    'valid': bool(valid),
                    'path': path,
                }
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    return pseudo_bank, {
        'valid_count': valid_count,
        'total_count': len(pseudo_bank),
        'valid_ratio': float(valid_count / max(1, len(pseudo_bank))),
        'mean_confidence': mean_conf,
    }


def extract_features(model, loader, device):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            _, embedding = model(imgs)
            all_features.append(embedding)
            all_labels.append(labels)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def evaluate_model(model, loader, device, distance_metric='cosine', normalize=True):
    features, labels = extract_features(model, loader, device)
    metrics = compute_metrics(features, labels, distance_metric=distance_metric, normalize=normalize)
    return metrics, features.detach().cpu().numpy(), labels.detach().cpu().numpy()


def compute_pair_scores(features, labels, distance_metric='cosine', normalize=True):
    feats = np.asarray(features, dtype=np.float32)
    labs = np.asarray(labels)
    if normalize:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.maximum(norms, 1e-12)
    if distance_metric == 'cosine':
        score_matrix = feats @ feats.T
    else:
        score_matrix = -cdist(feats, feats, metric='euclidean')
    positive_scores = []
    negative_scores = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            if labs[i] == labs[j]:
                positive_scores.append(float(score_matrix[i, j]))
            else:
                negative_scores.append(float(score_matrix[i, j]))
    return positive_scores, negative_scores


def save_verification_plots(features, labels, save_dir, prefix, distance_metric='cosine', normalize=True):
    os.makedirs(save_dir, exist_ok=True)
    positive_scores, negative_scores = compute_pair_scores(features, labels, distance_metric=distance_metric, normalize=normalize)
    if not positive_scores or not negative_scores:
        return
    y_true = np.array([1] * len(positive_scores) + [0] * len(negative_scores))
    y_score = np.array(positive_scores + negative_scores)
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    fnr = 1.0 - tpr

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FAR')
    plt.ylabel('TAR')
    plt.title(f'{prefix} ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_roc.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, fnr)
    plt.xlabel('FAR')
    plt.ylabel('FRR')
    plt.title(f'{prefix} DET')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_det.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.hist(negative_scores, bins=100, alpha=0.6, label='different')
    plt.hist(positive_scores, bins=100, alpha=0.6, label='same')
    plt.xlabel('score')
    plt.ylabel('count')
    plt.title(f'{prefix} score distribution')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_score_distribution.png'), dpi=200)
    plt.close()


def save_domain_tsne_plot(
    source_features,
    target_features,
    save_path,
    title,
    normalize=True,
    random_state=42,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    source_features = np.asarray(source_features, dtype=np.float32)
    target_features = np.asarray(target_features, dtype=np.float32)
    if len(source_features) == 0 or len(target_features) == 0:
        return

    features = np.concatenate([source_features, target_features], axis=0)
    domain_labels = np.concatenate([
        np.zeros(len(source_features), dtype=np.int32),
        np.ones(len(target_features), dtype=np.int32),
    ])

    if normalize:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)

    perplexity = min(30, max(5, (len(features) - 1) // 3))
    embedding_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        random_state=random_state,
    ).fit_transform(features)

    title_font = get_plot_font(size=15)
    legend_font = get_plot_font(size=12)
    legend_title_font = get_plot_font(size=13)

    plt.figure(figsize=(7, 6))
    source_mask = domain_labels == 0
    target_mask = domain_labels == 1
    plt.scatter(
        embedding_2d[source_mask, 0],
        embedding_2d[source_mask, 1],
        s=12,
        alpha=0.75,
        c='#1f77b4',
        label='源域验证集',
    )
    plt.scatter(
        embedding_2d[target_mask, 0],
        embedding_2d[target_mask, 1],
        s=12,
        alpha=0.75,
        c='#d62728',
        label='目标域验证集',
    )
    plt.title(title, fontproperties=title_font)
    legend = plt.legend(loc='best', prop=legend_font, frameon=True, title='图示说明')
    legend.get_title().set_fontproperties(legend_title_font)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def sample_shared_identities(source_labels, target_labels, num_identities=10, random_state=42):
    source_ids = set(np.asarray(source_labels).tolist())
    target_ids = set(np.asarray(target_labels).tolist())
    shared_ids = sorted(source_ids & target_ids)
    if not shared_ids:
        return []
    if len(shared_ids) <= num_identities:
        return shared_ids
    rng = np.random.default_rng(random_state)
    selected = rng.choice(shared_ids, size=num_identities, replace=False)
    return sorted(int(x) for x in selected.tolist())


def save_identity_tsne_plot(
    source_features,
    source_labels,
    target_features,
    target_labels,
    selected_ids,
    save_path,
    title,
    normalize=True,
    random_state=42,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    source_features = np.asarray(source_features, dtype=np.float32)
    target_features = np.asarray(target_features, dtype=np.float32)
    source_labels = np.asarray(source_labels, dtype=np.int32)
    target_labels = np.asarray(target_labels, dtype=np.int32)
    selected_ids = [int(x) for x in selected_ids]
    if len(selected_ids) == 0:
        return

    source_mask = np.isin(source_labels, selected_ids)
    target_mask = np.isin(target_labels, selected_ids)
    if not source_mask.any() or not target_mask.any():
        return

    src_feats = source_features[source_mask]
    src_labs = source_labels[source_mask]
    tgt_feats = target_features[target_mask]
    tgt_labs = target_labels[target_mask]

    features = np.concatenate([src_feats, tgt_feats], axis=0)
    domain_labels = np.concatenate([
        np.zeros(len(src_feats), dtype=np.int32),
        np.ones(len(tgt_feats), dtype=np.int32),
    ])
    identity_labels = np.concatenate([src_labs, tgt_labs], axis=0)

    if normalize:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)

    perplexity = min(20, max(5, (len(features) - 1) // 3))
    embedding_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        random_state=random_state,
    ).fit_transform(features)

    colors = plt.cm.get_cmap('tab10', len(selected_ids))
    title_font = get_plot_font(size=17)
    legend_font = get_plot_font(size=13)
    legend_title_font = get_plot_font(size=14)
    plt.figure(figsize=(9, 7))

    src_embed = embedding_2d[:len(src_feats)]
    tgt_embed = embedding_2d[len(src_feats):]

    for idx, identity in enumerate(selected_ids):
        color = colors(idx)
        src_id_mask = src_labs == identity
        tgt_id_mask = tgt_labs == identity
        if not src_id_mask.any() or not tgt_id_mask.any():
            continue

        src_points = src_embed[src_id_mask]
        tgt_points = tgt_embed[tgt_id_mask]
        src_center = src_points.mean(axis=0)
        tgt_center = tgt_points.mean(axis=0)

        plt.scatter(
            src_points[:, 0],
            src_points[:, 1],
            s=32,
            alpha=0.78,
            marker='o',
            color=color,
            edgecolors='none',
        )
        plt.scatter(
            tgt_points[:, 0],
            tgt_points[:, 1],
            s=38,
            alpha=0.82,
            marker='^',
            color=color,
            edgecolors='none',
        )
        plt.scatter(
            src_center[0],
            src_center[1],
            s=120,
            marker='X',
            color=color,
            edgecolors='black',
            linewidths=0.7,
        )
        plt.scatter(
            tgt_center[0],
            tgt_center[1],
            s=130,
            marker='P',
            color=color,
            edgecolors='black',
            linewidths=0.7,
        )
        plt.plot(
            [src_center[0], tgt_center[0]],
            [src_center[1], tgt_center[1]],
            linestyle='--',
            linewidth=1.0,
            alpha=0.8,
            color=color,
        )

    domain_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='源域样本'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='目标域样本'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=10, label='源域中心'),
        plt.Line2D([0], [0], marker='P', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=10, label='目标域中心'),
    ]
    legend_domain = plt.legend(
        handles=domain_handles,
        loc='upper right',
        frameon=True,
        title='图示说明',
        prop=legend_font,
    )
    legend_domain.get_title().set_fontproperties(legend_title_font)
    plt.gca().add_artist(legend_domain)

    plt.title(title, fontproperties=title_font)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_pseudo_snapshot(pseudo_bank, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'pseudo_label', 'confidence', 'valid', 'path'])
        for idx in sorted(pseudo_bank.keys()):
            row = pseudo_bank[idx]
            writer.writerow([idx, row['pseudo_label'], f"{row['confidence']:.6f}", int(row['valid']), row['path']])


def append_jsonl(record, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
