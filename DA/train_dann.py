
from math import e
import os
from re import L
from losses.arcface import ArcFaceLoss
import torch
import torch.optim as optim
import numpy as np
import logging
from torch.nn import functional as F
import torch.nn as nn
from models.resnet import ResNet_double, BasicBlock
from losses import TripletLoss, MMDLoss
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, average_precision_score
from val_base import validate, compute_metrics 
from tqdm import tqdm
from DA.DA import write_info
import sys
sys.path.insert(0, '/home/workstation/Palm/PR_20241226_copy')
from utils.grad_reverse import grad_reverse


class DANNModel(nn.Module):
    def __init__(self, base_model, num_classes=150, feature_dim=128):
        super(DANNModel, self).__init__()
        # 基础特征提取器（复用原ResNet_double的特征提取部分）
        self.feature_extractor = base_model  # 输出：(conv_features, embedding)
        self.feature_dim = feature_dim
        
        # 标签分类头（用于源域有监督学习）
        self.label_classifier = ArcFaceLoss(
            embedding_size=feature_dim,
            class_num=num_classes,
            s=64,
            m=0.50,
            gamma=2
        )
        
        # 域分类头（用于域对抗，区分源域/目标域）
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            # nn.Sigmoid()  # 二分类：源域0，目标域1
        )
        
    def forward(self, x, lambd=1.0):
        # 特征提取
        conv_feats, embedding = self.feature_extractor(x)  # embedding: [batch, 128]
        # 梯度反转（仅用于域分类器）
        reversed_embedding = grad_reverse(embedding, lambd=lambd)
        # 域分类输出
        domain_pred = self.domain_classifier(reversed_embedding)  # [batch, 1]
        return conv_feats, embedding, domain_pred

def train_teachers_dann(
    source_loader, source_val_loader,
    target_train_loaders, target_val_loaders,
    num_teachers, epochs, lr, writer, model_save_path,
    domain_loss_weight=0.1  # 域损失权重，可动态调整
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(model_save_path, exist_ok=True)
    
    # 初始化DANN教师模型（每个教师都是DANN结构）
    base_models = [ResNet_double(BasicBlock, [2,2,2,2]).to(device) for _ in range(num_teachers)]
    teachers = [DANNModel(base_model, num_classes=150).to(device) for base_model in base_models]
    optimizers = [optim.Adam(t.parameters(), lr=lr, eps=1e-08) for t in teachers]
    # schedulers = [optim.lr_scheduler.StepLR(optimizers[i], step_size=10, gamma=0.1) for i in range(num_teachers)]
    schedulers = [optim.lr_scheduler.MultiStepLR(optimizers[i], milestones=[20,60,100,140], gamma=0.2) for i in range(num_teachers)]
    
    # 损失函数
    arcface_loss = ArcFaceLoss(embedding_size=128, class_num=150, s=64, m=0.50, gamma=2).to(device)
    domain_criterion = nn.BCEWithLogitsLoss().to(device)  # 域分类损失（二分类）
    
    # 混合精度训练器
    scalers = [torch.amp.GradScaler() for _ in range(num_teachers)]

    best_eer = [float('inf') for _ in range(num_teachers)]
    for epoch in tqdm(range(epochs)):
        for i, teacher in enumerate(teachers):
            optimizer = optimizers[i]
            scheduler = schedulers[i]
            scaler = scalers[i]
            teacher.train()
            epoch_loss = 0.0
            epoch_label_loss = 0.0
            epoch_domain_loss = 0.0

            src_iter = iter(source_loader)
            tgt_iter = iter(target_train_loaders[i])
            steps = min(len(source_loader), len(target_train_loaders[i]))

            # 动态调整域损失权重（随epoch增加，增强对抗性）
            current_domain_weight = domain_loss_weight * min(1.0, (epoch + 1) / 10)

            for _ in range(steps):
                optimizer.zero_grad()
                # 加载源域和目标域数据
                try:
                    s_imgs, s_labels = next(src_iter)
                except StopIteration:
                    src_iter = iter(source_loader)
                    s_imgs, s_labels = next(src_iter)
                try:
                    t_imgs, _ = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(target_train_loaders[i])
                    t_imgs, _ = next(tgt_iter)

                s_imgs, s_labels = s_imgs.to(device), s_labels.to(device)
                t_imgs = t_imgs.to(device)
                batch_size = s_imgs.size(0)

                # 混合精度前向传播
                with torch.amp.autocast(device_type='cuda'):
                    # 源域数据：计算标签损失和域损失
                    _, s_embedding, s_domain_pred = teacher(s_imgs)
                    label_loss = arcface_loss(s_embedding, s_labels)  # 标签分类损失
                    s_domain_label = torch.zeros(batch_size, 1, device=device)  # 源域标签0
                    s_domain_loss = domain_criterion(s_domain_pred, s_domain_label)

                    # 目标域数据：仅计算域损失（无标签）
                    _, t_embedding, t_domain_pred = teacher(t_imgs)
                    t_domain_label = torch.ones(batch_size, 1, device=device)  # 目标域标签1
                    t_domain_loss = domain_criterion(t_domain_pred, t_domain_label)

                    # 总损失：标签损失 + 域损失（源+目标）
                    total_domain_loss = (s_domain_loss + t_domain_loss) # / 2
                    total_loss = label_loss + current_domain_weight * total_domain_loss

                # 混合精度反向传播
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += total_loss.item()
                epoch_label_loss += label_loss.item()
                epoch_domain_loss += total_domain_loss.item()

            # 记录损失
            avg_loss = epoch_loss / steps
            avg_label_loss = epoch_label_loss / steps
            avg_domain_loss = epoch_domain_loss / steps
            writer.add_scalar(f'teacher_{i}/total_loss', avg_loss, epoch)
            writer.add_scalar(f'teacher_{i}/label_loss', avg_label_loss, epoch)
            writer.add_scalar(f'teacher_{i}/domain_loss', avg_domain_loss, epoch)
            writer.add_scalar(f'teacher_{i}/lr', scheduler.get_last_lr()[0], epoch)
            print(f'Teacher {i} Epoch {epoch}: Total Loss={avg_loss:.4f}, Label Loss={avg_label_loss:.4f}, Domain Loss={avg_domain_loss:.4f}')

            scheduler.step()

            # 验证
            metrics = validate(teacher.feature_extractor, target_val_loaders[i])  # 用特征提取器验证
            write_info(avg_loss, metrics, writer, epoch, prefix=f'teacher_{i}_target')
            if metrics['eer'] < best_eer[i]:
                best_eer[i] = metrics['eer']
                torch.save(teacher.state_dict(), os.path.join(model_save_path, f'teacher_{i}_best.pth'))

    return teachers


def train_student_dann(
    teachers, source_loader, target_train_loaders,
    student_val_loader, epochs, lr, writer, model_save_path,
    domain_loss_weight=0.1, distill_weight=0.5  # 蒸馏损失权重
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(model_save_path, exist_ok=True)

    # 学生模型（DANN结构）
    student_base = ResNet_double(BasicBlock, [2,2,2,2]).to(device)
    student = DANNModel(student_base, num_classes=150).to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # 损失函数
    arcface_loss = ArcFaceLoss(embedding_size=128, class_num=150, s=64, m=0.50, gamma=2).to(device)
    domain_criterion = nn.BCEWithLogitsLoss().to(device)
    distill_criterion = nn.MSELoss().to(device)  # 蒸馏损失（学生特征模仿教师）

    scaler = torch.amp.GradScaler()
    best_eer = float('inf')
    tgt_iters = [iter(loader) for loader in target_train_loaders]

    for epoch in tqdm(range(epochs)):
        student.train()
        epoch_loss = 0.0
        current_domain_weight = domain_loss_weight * min(1.0, (epoch + 1) / 10)

        for s_imgs, s_labels in source_loader:
            optimizer.zero_grad()
            s_imgs, s_labels = s_imgs.to(device), s_labels.to(device)
            batch_size = s_imgs.size(0)

            # 加载目标域数据
            t_imgs_list = []
            for i, t_iter in enumerate(tgt_iters):
                try:
                    t_imgs, _ = next(t_iter)
                except StopIteration:
                    tgt_iters[i] = iter(target_train_loaders[i])
                    t_imgs, _ = next(tgt_iters[i])
                t_imgs_list.append(t_imgs.to(device))
            t_imgs = torch.cat(t_imgs_list, dim=0)  # 合并多目标域数据

            # 混合精度前向
            with torch.amp.autocast(device_type='cuda'):
                # 学生对源域的输出
                _, s_emb, s_domain_pred = student(s_imgs)
                label_loss = arcface_loss(s_emb, s_labels)
                s_domain_loss = domain_criterion(s_domain_pred, torch.zeros(batch_size, 1, device=device))

                # 学生对目标域的输出
                _, t_emb, t_domain_pred = student(t_imgs)
                t_domain_loss = domain_criterion(t_domain_pred, torch.ones(t_imgs.size(0), 1, device=device))
                domain_loss = (s_domain_loss + t_domain_loss) / 2

                # 蒸馏损失（学生模仿所有教师的特征）
                distill_loss = 0.0
                for teacher in teachers:
                    teacher.eval()
                    with torch.no_grad():
                        _, t_emb_teacher, _ = teacher(t_imgs)
                    distill_loss += distill_criterion(t_emb, t_emb_teacher)
                distill_loss /= len(teachers)

                # 总损失
                total_loss = label_loss + current_domain_weight * domain_loss + distill_weight * distill_loss

            # 反向传播
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()

        # 记录与验证
        avg_loss = epoch_loss / len(source_loader)
        writer.add_scalar('student/total_loss', avg_loss, epoch)
        writer.add_scalar('student/lr', scheduler.get_last_lr()[0], epoch)
        print(f'Student Epoch {epoch}: Loss={avg_loss:.4f}')

        scheduler.step()
        metrics = validate(student.feature_extractor, student_val_loader)
        write_info(avg_loss, metrics, writer, epoch, prefix='student')
        if metrics['eer'] < best_eer:
            best_eer = metrics['eer']
            torch.save(student.state_dict(), os.path.join(model_save_path, 'student_best.pth'))

    return student