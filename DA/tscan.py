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
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader

from configs import settings
from losses.arcface import ArcFaceLoss
from models.resnet import ResNet_double, BasicBlock
from utils.grad_reverse import grad_reverse
from utils.utils import list_pictures
from val_base import compute_metrics


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


def build_loaders(source_root, target_root, batch_size, num_workers, use_augmentation=True):
    light_transform = build_light_train_transform() if use_augmentation else build_eval_transform()
    source_train = PalmSourceDataset(source_root, light_transform, split='train')
    source_proto = PalmSourceDataset(source_root, build_eval_transform(), split='train')
    source_test = PalmEvalDataset(source_root, build_eval_transform(), split='test')
    target_train = PalmTargetDataset(target_root, light_transform, light_transform, split='train')
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
