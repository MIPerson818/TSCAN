import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_curve, average_precision_score
from scipy.spatial.distance import cdist


def compute_metrics(features, labels, distance_metric="cosine", normalize=True, far_levels=(0.1, 0.01, 0.001)):
    feats = features.detach().cpu().numpy().astype(np.float32)
    labs = labels.detach().cpu().numpy()

    if normalize:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.maximum(norms, 1e-12)

    num_samples = len(labs)
    if num_samples < 2:
        return {
            "eer": 0.0,
            "acc": 0.0,
            "true_acc": 0.0,
            "mAP": 0.0,
            "tar_at_far": {far: 0.0 for far in far_levels},
            "fpr": np.array([]),
            "tpr": np.array([]),
            "fnr": np.array([]),
            "thresholds": np.array([]),
        }

    if distance_metric == "cosine":
        score_matrix = feats @ feats.T
    elif distance_metric == "euclidean":
        distance_matrix = cdist(feats, feats, metric="euclidean")
        score_matrix = -distance_matrix
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    scores, targets = [], []
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            scores.append(score_matrix[i, j])
            targets.append(int(labs[i] == labs[j]))
    scores = np.asarray(scores)
    targets = np.asarray(targets)

    fpr, tpr, thresholds = roc_curve(targets, scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    mAP = float(average_precision_score(targets, scores))

    eer_threshold = thresholds[eer_idx]
    true_scores = scores[targets == 1]
    false_scores = scores[targets == 0]
    tp = float((true_scores >= eer_threshold).sum())
    fp = float((false_scores >= eer_threshold).sum())
    tn = float(len(false_scores) - fp)
    fn = float(len(true_scores) - tp)
    acc = float((tp + tn) / max(1.0, tp + tn + fp + fn))

    if distance_metric == "cosine":
        nn_matrix = score_matrix.copy()
        np.fill_diagonal(nn_matrix, -np.inf)
        nearest_idx = np.argmax(nn_matrix, axis=1)
    else:
        distance_matrix = cdist(feats, feats, metric="euclidean")
        np.fill_diagonal(distance_matrix, np.inf)
        nearest_idx = np.argmin(distance_matrix, axis=1)
    true_acc = float((labs[nearest_idx] == labs).mean())

    tar_at_far = {}
    for far in far_levels:
        valid = np.where(fpr <= far)[0]
        tar_at_far[far] = 0.0 if len(valid) == 0 else float(tpr[valid].max())

    return {
        "eer": eer,
        "acc": acc,
        "true_acc": true_acc,
        "mAP": mAP,
        "tar_at_far": tar_at_far,
        "fpr": fpr,
        "tpr": tpr,
        "fnr": fnr,
        "thresholds": thresholds,
    }


def validate(model, val_loader, distance_metric="cosine", normalize=True, far_levels=(0.1, 0.01, 0.001)):
    device = next(model.parameters()).device
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            _, feats = model(imgs)
            all_feats.append(feats)
            all_labels.append(labels)
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return compute_metrics(all_feats, all_labels, distance_metric=distance_metric, normalize=normalize, far_levels=far_levels)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='evaluate a model on a dataset')
    parser.add_argument('--model', type=str, default='resnet18', help='model architecture (default: resnet18)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name (default: cifar10)')
    parser.add_argument('--data-dir', type=str, default='./data', help='path to dataset (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers (default: 4)')


if __name__ == '__main__':
    pass
