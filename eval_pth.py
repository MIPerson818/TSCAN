#!/usr/bin/env python3

"""
Test face recognition model performance
Calculate EER, TAR@FAR and generate DET/ROC curves
Author: Modified from original script
"""
import sys
import sklearn
sys.path.insert(0, "/home/workstation/Palm/PR_20241226_copy")  # 确认路径正确（建议使用绝对路径）
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from utils.utils import get_network, get_test_dataloader
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import time
import math
from scipy.spatial.distance import cdist
from configs import settings
from models.resnet import ResNet_double, BasicBlock

# 设置字体支持（保持英文为主，避免中文显示问题）
# plt.rcParams["font.family"] = ["Times New Roman", "Arial"]
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def main():
    parser = argparse.ArgumentParser(description='Face Recognition Evaluation')

    parser.add_argument('-num_classes', type=int, default=100, help='Number of classes')
    parser.add_argument('-net', type=str, default='resnet50', help='Network architecture')
    parser.add_argument('-gpu', type=str, default='0', help='GPU device IDs')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--source_path', type=str, nargs='+', 
                        default=['/home/repository/PalmDatas/XJTUUP/iPhone/Nature/cent_test'],   # Nature Flash HUAWEI iPhone LG MI8 Samsung
                        help='Path to source dataset')
    parser.add_argument('--model_path', type=str, 
                        default='/home/workstation/Palm/PR_20241226_copy/results/DA_distill_V0.0.1/model/adapt/teachers/teacher_0.pth',
                        help='Path to pretrained model')
    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'],  # 修正拼写错误（consine→cosine）
                        help='Distance metric (cosine or euclidean)')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/home/workstation/Palm/PR_20241226_copy/tmp2/exp001',
        help='Directory to save evaluation artifacts',
    )
    
    args = parser.parse_args()
    
    # ------------ 关键修改1：统一设备管理 ------------
    # 根据参数和GPU可用性确定设备
    device = torch.device(
        f"cuda:{args.gpu}" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    # 创建评估结果保存目录
    eval_dir = args.save_dir
    os.makedirs(eval_dir, exist_ok=True)
    
    # 记录评估配置
    config_file = os.path.join(eval_dir, 'evaluation_config.txt')
    with open(config_file, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.source_path}\n")
        f.write(f"Distance: {args.distance}\n")
        f.write(f"Feature Dimension: {args.feature_dim}\n")
        f.write(f"Device: {device}\n")  # 记录使用的设备
    
    # 加载模型
    # net = get_network(args)
    net = ResNet_double(BasicBlock, [2,2,2,2])
    # ------------ 关键修改2：安全加载权重并移动到目标设备 ------------
    # 添加 weights_only=True 解决安全警告，同时指定设备
    state = torch.load(args.model_path, weights_only=True, map_location=device)
    if isinstance(state, dict) and 'backbone_state' in state:
        state = state['backbone_state']
    net.load_state_dict(state)
    net = net.to(device)  # 将模型移动到目标设备
    net.eval()  # 切换到评估模式
    
    # 加载数据
    data_source_loader = get_test_dataloader(
        args.source_path,
        settings.DATA_TRAIN_MEAN,
        settings.DATA_TRAIN_STD,
        num_workers=4,
        batch_size=64,
        class_number=args.num_classes,
        num_per=10,
    )
    
    # 提取特征
    print("Extracting features...")
    t0 = time.time()
    # ------------ 关键修改3：在目标设备上初始化特征张量 ------------
    source = torch.zeros(1, args.feature_dim, device=device)  # 直接在device上创建，避免后续移动
    s_label = torch.tensor([], dtype=torch.long, device=device)  # 标签也在device上初始化
    
    with torch.no_grad():
        for n_iter, (image, label) in tqdm(enumerate(data_source_loader)):
            # 将输入数据移动到目标设备
            image = image.to(device)
            label = label.to(device)
            
            # ------------ 关键修改4：确保模型输出与设备匹配 ------------
            # 注意：ResNet_double的forward返回(conv_features, feature_embedding2)
            # 根据之前的模型定义，第二个返回值是最终特征向量
            _, feat = net(image)  # 只取特征向量部分
            
            source = torch.cat((source, feat))
            s_label = torch.cat((s_label, label))
    
    # 处理特征（移到CPU并转为numpy）
    source = source[1:, :].cpu().numpy()  # 移除初始全零行并移到CPU
    s_label = s_label.cpu().numpy()  # 标签也移到CPU
    
    # 特征归一化
    result_code = np.reshape(source, [-1, args.feature_dim])
    result_code /= np.maximum(1e-5, np.linalg.norm(result_code, axis=1, keepdims=True))
    
    print(f"Features shape: {result_code.shape}")
    print(f"Extraction time: {time.time() - t0:.2f}s")
    
    # 构建人脸对并计算相似度/距离（修正拼写错误：consine→cosine）
    print("Calculating pairwise similarities/distances...")
    true_list = []  # 正样本对分数
    false_list = []  # 负样本对分数
    
    for i in tqdm(range(len(result_code))):
        for j in range(i + 1, len(result_code)):
            if s_label[i] == s_label[j]:
                if args.distance == 'cosine':
                    true_list.append(np.dot(result_code[i], result_code[j]))  # 余弦相似度
                else:  # euclidean
                    true_list.append(np.sqrt(np.sum(np.square(result_code[i] - result_code[j]))))  # 欧氏距离
            else:
                if args.distance == 'cosine':
                    false_list.append(np.dot(result_code[i], result_code[j]))
                else:  # euclidean
                    false_list.append(np.sqrt(np.sum(np.square(result_code[i] - result_code[j]))))
    
    print(f"Positive pairs: {len(true_list)}")
    print(f"Negative pairs: {len(false_list)}")
    
    # 保存原始分数数据
    scores_data = {
        'true_scores': true_list,
        'false_scores': false_list
    }
    np.save(os.path.join(eval_dir, 'scores_data.npy'), scores_data)
    
    # 计算评估指标
    print("\nCalculating evaluation metrics...")
    metrics = calculate_metrics(true_list, false_list, args.distance)
    metrics['true_acc'] = calculate_true_acc(result_code, s_label, args.distance)
    
    # 保存指标到文件
    metrics_file = os.path.join(eval_dir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        # 保存主要指标
        for key in ['eer', 'auc', 'acc', 'true_acc']:
            if key in metrics:
                f.write(f"{key}: {metrics[key]:.6f}\n")
        
        # 保存 TAR@FAR
        f.write("\nTAR@FAR:\n")
        for far, tar in metrics['tar_at_far'].items():
            f.write(f"TAR@FAR={far}: {tar:.6f}\n")

    # 打印指标
    print("\nEvaluation Results:\n")
    print(f"ACC: {metrics['acc']:.6f}")  # 新增 ACC 打印
    print(f"TrueACC: {metrics['true_acc']:.6f}")
    print(f"EER: {metrics['eer']:.6f}")
    # print(f"AUC: {metrics['auc']:.6f}")
    print("\nTAR@FAR:")
    for far, tar in metrics['tar_at_far'].items():
        print(f"TAR@FAR={far}: {tar:.6f}")
    
    # 绘制并保存图表
    print("\nGenerating plots...")
    plot_det_curve(true_list, false_list, args.distance, eval_dir)
    plot_roc_curve(true_list, false_list, args.distance, eval_dir)
    plot_score_distribution(true_list, false_list, args.distance, eval_dir)
    
    print(f"\nAll results saved to: {eval_dir}")
    print(f"Parameter numbers: {sum(p.numel() for p in net.parameters())}")
    print(f"Total evaluation time: {(time.time() - t0) / 60:.2f} minutes")


def calculate_true_acc(features, labels, distance_metric):
    """识别准确率：LOOCV 1-NN accuracy"""
    if len(labels) < 2:
        return 0.0

    if distance_metric == 'cosine':
        similarity = np.matmul(features, features.T)
        np.fill_diagonal(similarity, -np.inf)
        nearest_idx = np.argmax(similarity, axis=1)
    else:
        distance_matrix = cdist(features, features, metric='euclidean')
        np.fill_diagonal(distance_matrix, np.inf)
        nearest_idx = np.argmin(distance_matrix, axis=1)

    pred_labels = labels[nearest_idx]
    return float((pred_labels == labels).mean())


def calculate_metrics(true_list, false_list, distance_metric):
    """计算评估指标：EER、AUC、ACC和TAR@FAR（按定义修正）"""
    # 基础参数
    total_true = len(true_list)  # 正样本对总数（同一手掌）
    total_false = len(false_list)  # 负样本对总数（不同手掌）
    if total_true == 0 or total_false == 0:
        return {"eer": 0.0, "auc": 0.0, "acc": 0.0, "tar_at_far": {}}

    # ----------------------------
    # 1. 计算FPR、TPR、AUC（用于EER和AUC）
    # ----------------------------
    y_true = [1] * total_true + [0] * total_false  # 1:正样本对，0:负样本对
    if distance_metric == 'cosine':
        # 余弦相似度：分数越高越可能匹配
        y_score = true_list + false_list
    else:  # euclidean
        # 欧氏距离：分数越低越可能匹配（转为负距离以便统一ROC逻辑）
        y_score = [-d for d in true_list] + [-d for d in false_list]

    # 计算ROC曲线（FPR=FP/(FP+TN)=FP/总负样本数；TPR=TP/总正样本数=TAR）
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr  # FRR = 1 - TAR
    auc = sklearn.metrics.auc(fpr, tpr)

    # 计算EER（FAR=FRR时的阈值）
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]  # 此时FAR=FRR=eer


    # ----------------------------
    # 2. 计算ACC（基于EER阈值）
    # ----------------------------
    # 获取EER对应的原始距离阈值（还原欧氏距离的符号）
    if distance_metric == 'cosine':
        eer_threshold = thresholds[eer_idx]
    else:
        eer_threshold = -thresholds[eer_idx]  # 还原为原始欧氏距离

    # 计算TP、TN、FP、FN（基于EER阈值）
    if distance_metric == 'cosine':
        # 余弦相似度：>=阈值为匹配（正样本正确接受，负样本错误接受）
        TP = sum(1 for s in true_list if s >= eer_threshold)
        FP = sum(1 for s in false_list if s >= eer_threshold)
    else:  # euclidean
        # 欧氏距离：<=阈值为匹配
        TP = sum(1 for s in true_list if s <= eer_threshold)
        FP = sum(1 for s in false_list if s <= eer_threshold)

    TN = total_false - FP  # 负样本正确拒绝（不匹配）
    FN = total_true - TP  # 正样本错误拒绝（不匹配）

    # 计算ACC：(正确分类的样本对) / 总样本对
    acc = (TP + TN) / (total_true + total_false) if (total_true + total_false) > 0 else 0.0


    # ----------------------------
    # 3. 修正TAR@FAR计算（按定义：FAR=FP/总负样本数，TAR=TP/总正样本数）
    # ----------------------------
    tar_at_far = {}
    far_levels = [0.1, 0.01, 0.001, 0.0001, 0.00001]  # 目标FAR值

    # 对负样本排序，用于找阈值
    if distance_metric == 'cosine':
        # 余弦相似度：负样本按降序排序（高分易被错误接受）
        false_sorted = sorted(false_list, reverse=True)
    else:  # euclidean
        # 欧氏距离：负样本按升序排序（低分易被错误接受）
        false_sorted = sorted(false_list, reverse=False)

    for far in far_levels:
        # 计算目标FAR对应的负样本错误接受数量（FP = FAR * 总负样本数）
        fp_target = int(far * total_false)
        if fp_target <= 0 or fp_target >= total_false:
            continue  # 避免极端值

        # 找到对应阈值（第fp_target个负样本的分数）
        threshold = false_sorted[fp_target - 1]  # 索引从0开始

        # 计算该阈值下的TAR（正确接受率=TP/总正样本数）
        if distance_metric == 'cosine':
            TP_current = sum(1 for s in true_list if s >= threshold)
        else:
            TP_current = sum(1 for s in true_list if s <= threshold)
        tar = TP_current / total_true if total_true > 0 else 0.0

        # 记录TAR@FAR
        tar_at_far[far] = tar


    return {
        'eer': eer,
        'auc': auc,
        'acc': acc,  # 新增ACC指标
        'tar_at_far': tar_at_far,
        'fpr': fpr,
        'tpr': tpr,
        'fnr': fnr,
        'thresholds': thresholds
    }


def plot_det_curve(true_list, false_list, distance_metric, save_dir):
    """绘制DET曲线（FAR vs FRR）"""
    # 准备标签和分数
    y_true = [1] * len(true_list) + [0] * len(false_list)
    if distance_metric == 'cosine':
        y_score = true_list + false_list
    else:  # euclidean
        y_score = [-d for d in true_list] + [-d for d in false_list]
    
    # 计算FPR, TPR和阈值
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr  # 错误拒绝率（FRR）
    
    # 创建DET曲线图表
    plt.figure(figsize=(10, 8))
    
    # 使用对数坐标轴（适合展示小概率）
    plt.semilogx(fpr, fnr, 'b-', linewidth=2)
    
    # 找到EER点
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    
    # 标记EER点
    plt.plot(fpr[eer_index], fnr[eer_index], 'ro', markersize=8)
    plt.annotate(f'EER = {eer:.6f}', 
                 xy=(fpr[eer_index], fnr[eer_index]), 
                 xytext=(fpr[eer_index] * 3, fnr[eer_index] * 3),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # 设置坐标轴范围和标签（英文更规范）
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('False Acceptance Rate (FAR)', fontsize=14)
    plt.ylabel('False Rejection Rate (FRR)', fontsize=14)
    plt.title('Detection Error Tradeoff (DET) Curve', fontsize=16)
    
    # 设置坐标轴刻度范围
    plt.xlim([1e-5, 0.1])
    plt.ylim([1e-5, 0.1])
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'det_curve.png'), dpi=300)
    plt.close()


def plot_roc_curve(true_list, false_list, distance_metric, save_dir):
    """绘制ROC曲线（TPR vs FPR）"""
    # 准备标签和分数
    y_true = [1] * len(true_list) + [0] * len(false_list)
    if distance_metric == 'cosine':
        y_score = true_list + false_list
    else:  # euclidean
        y_score = [-d for d in true_list] + [-d for d in false_list]
    
    # 计算FPR, TPR和AUC
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # 创建ROC曲线图表
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.6f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)  # 随机猜测基线
    
    # 设置坐标轴和标签
    plt.grid(True, alpha=0.5)
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
    plt.close()


def plot_score_distribution(true_list, false_list, distance_metric, save_dir):
    """绘制正负样本分数分布直方图"""
    plt.figure(figsize=(12, 8))
    
    # 确定距离类型的标签（英文）
    if distance_metric == 'cosine':
        distance_label = 'Cosine Similarity'
        hist_range = (-1.0, 1.0)  # 余弦相似度范围固定
    else:
        distance_label = 'Euclidean Distance'
        # 动态确定欧氏距离的范围
        all_scores = true_list + false_list
        hist_range = (min(all_scores) * 0.9, max(all_scores) * 1.1)
    
    # 绘制直方图
    plt.hist(true_list, bins=50, alpha=0.5, label='Genuine Pairs', range=hist_range, density=True)
    plt.hist(false_list, bins=50, alpha=0.5, label='Impostor Pairs', range=hist_range, density=True)
    
    # 添加图例和标签
    plt.xlabel(distance_label, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Score Distribution', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'score_distribution.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
