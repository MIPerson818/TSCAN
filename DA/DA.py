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


def convolution_distillation_loss(teacher_conv, student_conv):
    return F.mse_loss(teacher_conv, student_conv)

def feature_distillation_loss(teacher_fea, student_fea):
    huber_loss = torch.nn.SmoothL1Loss(beta=1.0)
    Ldis_fea = huber_loss(teacher_fea, student_fea)
    Langle_fea = 1 - F.cosine_similarity(teacher_fea, student_fea, dim=1).mean()
    return Ldis_fea + Langle_fea

def write_info(loss, metrics, writer, epoch, prefix='model'):
    writer.add_scalar(f'{prefix}/val_EER', metrics['eer'], epoch)
    writer.add_scalar(f'{prefix}/val_accuracy', metrics['acc'], epoch)
    writer.add_scalar(f'{prefix}/val_TrueACC', metrics['true_acc'], epoch)
    writer.add_scalar(f'{prefix}/val_mAP', metrics['mAP'], epoch)
    writer.add_scalar(f'{prefix}/val_TAR@FAR_0.1', metrics['tar_at_far'].get(0.1, 0.0), epoch)
    writer.add_scalar(f'{prefix}/val_TAR@FAR_0.01', metrics['tar_at_far'].get(0.01, 0.0), epoch)
    writer.add_scalar(f'{prefix}/val_TAR@FAR_0.001', metrics['tar_at_far'].get(0.001, 0.0), epoch)
    logging.info(
        f"{prefix} Epoch {epoch}: EER={metrics['eer']:.4f}, ACC={metrics['acc']:.4f}, "
        f"TrueACC={metrics['true_acc']:.4f}, mAP={metrics['mAP']:.4f}, "
        f"TAR@0.1={metrics['tar_at_far'].get(0.1, 0.0):.4f}, "
        f"TAR@0.01={metrics['tar_at_far'].get(0.01, 0.0):.4f}, "
        f"TAR@0.001={metrics['tar_at_far'].get(0.001, 0.0):.4f}"
    )
    print(
        f"{prefix} Epoch {epoch}: Loss={loss:.4f}, EER={metrics['eer']:.4f}, ACC={metrics['acc']:.4f}, "
        f"TrueACC={metrics['true_acc']:.4f}, mAP={metrics['mAP']:.4f}, "
        f"TAR@0.1={metrics['tar_at_far'].get(0.1, 0.0):.4f}, "
        f"TAR@0.01={metrics['tar_at_far'].get(0.01, 0.0):.4f}, "
        f"TAR@0.001={metrics['tar_at_far'].get(0.001, 0.0):.4f}"
    )
    return None


def train_teachers(
    source_loader, source_val_loader,
    target_train_loaders, target_val_loaders,
    num_teachers, epochs, lr, writer, model_save_path
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(model_save_path, exist_ok=True)
    
    # 初始化教师模型和优化器
    teachers = [ResNet_double(BasicBlock, [2,2,2,2]).to(device) for _ in range(num_teachers)]
    optimizers = [optim.Adam(t.parameters(), lr=lr, eps=1e-08) for t in teachers]
    # schedulers = [optim.lr_scheduler.StepLR(optimizers[i], step_size=10, gamma=0.1) for i in range(num_teachers)]
    schedulers = [optim.lr_scheduler.MultiStepLR(optimizers[i], milestones=[20,60,100,140], gamma=0.2) for i in range(num_teachers)]

    
    # schedulers = [optim.lr_scheduler.CosineAnnealingWarmRestarts(   
    #     optimizers[i],
    #     T_0=10,        # 初始周期（10个epoch后第一次重启）
    #     T_mult=2,      # 每次重启后周期翻倍
    #     eta_min=1e-6   # 最小学习率
    # ) for i in range(num_teachers)]  # 前几个 epoch 学习率从较小值缓慢增长（热身），避免初期震荡；之后按余弦曲线衰减，平衡探索和收敛，通常比固定步长衰减收敛更快。

    # 损失函数
    arcface_loss = ArcFaceLoss(embedding_size=128, class_num=150, s=64, m=0.50, gamma=2).to(device)
    mmd_loss = MMDLoss(kernel_type='linear', kernel_mul=2.0, kernel_num=5, fix_sigma=None).to(device)

    # 初始化混合精度训练器（每个教师独立使用一个scaler，避免相互干扰）
    scalers = [torch.amp.GradScaler() for _ in range(num_teachers)]

    best_eer = [float('inf') for _ in range(num_teachers)]
    for epoch in tqdm(range(epochs)):
        for i, teacher in enumerate(teachers):
            optimizer = optimizers[i]
            scheduler = schedulers[i]
            scaler = scalers[i]  # 当前教师的scaler
            teacher.train()
            epoch_loss = 0.0

            src_iter = iter(source_loader)
            tgt_iter = iter(target_train_loaders[i])
            steps = min(len(source_loader), len(target_train_loaders[i]))

            for _ in range(steps):
                optimizer.zero_grad()
                # 安全获取源域和目标域batch
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

                # 混合精度前向传播（核心修改）
                with torch.amp.autocast(device_type='cuda'):  # 启用半精度计算
                    s_convs, s_feats = teacher(s_imgs)
                    t_convs, t_feats = teacher(t_imgs)
                    # 损失计算在半精度上下文内
                    loss_sup = arcface_loss(s_feats, s_labels)
                    loss_ada = mmd_loss(s_feats, t_feats)
                    loss = loss_sup + loss_ada

                # 混合精度反向传播（核心修改）
                scaler.scale(loss).backward()  # 缩放损失，避免梯度下溢
                scaler.step(optimizer)         # 根据缩放梯度更新参数
                scaler.update()                # 更新缩放器状态

                epoch_loss += loss.item()

            avg_loss = epoch_loss / steps
            writer.add_scalar(f'teacher_{i}/train_loss', avg_loss, epoch)
            writer.add_scalar(f'teacher_{i}/lr', scheduler.get_last_lr()[0], epoch)
            logging.info(f'teacher_{i} Epoch {epoch} train_loss={avg_loss:.4f}')
            print(f'teacher_{i} Epoch {epoch} train_loss={avg_loss:.4f}')

            scheduler.step()  # 学习率调度器在优化器更新后每个epoch执行

            # 验证（验证阶段无需混合精度，保持默认精度）
            metrics = validate(teacher, target_val_loaders[i])
            write_info(avg_loss, metrics, writer, epoch, prefix=f'teacher_{i}_target')
            if metrics['eer'] < best_eer[i]:
                best_eer[i] = metrics['eer']
                torch.save(teacher.state_dict(), os.path.join(model_save_path, f'teacher_{i}_best.pth'))
            metrics = validate(teacher, source_val_loader)
            write_info(avg_loss, metrics, writer, epoch, prefix=f'teacher_{i}_source')

            if epoch % 20 == 0:
                torch.save(teacher.state_dict(), os.path.join(model_save_path, f'teacher_{i}_epoch{epoch}.pth'))

    return teachers


def train_student(
    teachers, source_loader, target_train_loaders,
    student_val_loader, epochs, lr, writer, model_save_path, a=1.0, b=0.5
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(model_save_path, exist_ok=True)

    student = ResNet_double(BasicBlock, [2,2,2,2]).to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(   
        optimizer,
        T_0=10,        # 初始周期（10个epoch后第一次重启）
        T_mult=2,      # 每次重启后周期翻倍
        eta_min=1e-6   # 最小学习率
    )

    # 损失函数
    triplet_loss = TripletLoss(margin=0).to(device)
    arcface_loss = ArcFaceLoss(embedding_size=128, class_num=150, s=64, m=0.50, gamma=2).to(device)

    # 初始化混合精度训练器（学生模型单独的scaler）
    scaler = torch.amp.GradScaler()

    # 创建 target loader 迭代器
    tgt_iters = [iter(loader) for loader in target_train_loaders]

    best_eer = float('inf')
    for epoch in tqdm(range(epochs)):
        student.train()
        epoch_loss = 0.0

        # 源域有监督损失 + 蒸馏损失
        for s_imgs, s_labels in source_loader:
            optimizer.zero_grad()
            s_imgs, s_labels = s_imgs.to(device), s_labels.to(device)

            # 混合精度前向传播（核心修改）
            with torch.amp.autocast(device_type='cuda'):  # 启用半精度计算
                # 学生对源数据的前向
                s_convs_s, s_feats_s = student(s_imgs)
                loss_sup = arcface_loss(s_feats_s, s_labels)
                Ldis_conv = 0.0
                Ldis_fea = 0.0

                # 对每个目标域执行蒸馏
                for i, t_iter in enumerate(tgt_iters):
                    try:
                        t_imgs, _ = next(t_iter)
                    except StopIteration:
                        tgt_iters[i] = iter(target_train_loaders[i])
                        t_imgs, _ = next(tgt_iters[i])
                    t_imgs = t_imgs.to(device)
                    
                    # 教师模型前向（无梯度，半精度加速）
                    teacher = teachers[i]
                    teacher.eval()
                    with torch.no_grad():  # 教师不计算梯度
                        t_convs_t, t_feats_t = teacher(t_imgs)
                    
                    # 学生对目标数据的前向
                    t_convs_s, t_feats_s = student(t_imgs)
                    
                    # 蒸馏损失计算（半精度内）
                    Ldis_conv += convolution_distillation_loss(t_convs_t, t_convs_s)
                    Ldis_fea += feature_distillation_loss(t_feats_t, t_feats_s)

                # 总损失
                batch_loss = loss_sup + Ldis_fea * a + Ldis_conv * b

            # 混合精度反向传播（核心修改）
            scaler.scale(batch_loss).backward()  # 缩放损失
            scaler.step(optimizer)               # 更新参数
            scaler.update()                      # 更新缩放器
            epoch_loss += batch_loss.item()

        avg_loss = epoch_loss / len(source_loader)
        writer.add_scalar('student/train_loss', avg_loss, epoch)
        writer.add_scalar('student/lr', scheduler.get_last_lr()[0], epoch)
        logging.info(f'student Epoch {epoch} train_loss={avg_loss:.4f}')
        print(f'student Epoch {epoch} train_loss={avg_loss:.4f}')

        scheduler.step()  # 学习率调度器在优化器更新后每个epoch执行
        
        # 验证（保持默认精度）
        metrics = validate(student, student_val_loader)
        write_info(avg_loss, metrics, writer, epoch, prefix='student')
        if metrics['eer'] < best_eer:
            best_eer = metrics['eer']
            torch.save(student.state_dict(), os.path.join(model_save_path, 'student_best.pth'))

        if epoch % 20 == 0:
            torch.save(student.state_dict(), os.path.join(model_save_path, f'student_epoch{epoch}.pth'))

    return student

if __name__ == '__main__':
    pass