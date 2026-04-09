from math import e
import os
from re import L
from losses.arcface import ArcFaceLoss
from losses.adaface import AdaFaceLoss
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
import os
import logging
import shutil
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import pickle
import torch
from train_base import train
from utils.utils import list_pictures, Palmdata, get_all_image  # 你的原 utils
from configs import settings # 包含 DATA_TRAIN_MEAN, DATA_TRAIN_STD
import argparse
import yaml


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


def build_scheduler(optimizer, scheduler_name, scheduler_milestones, scheduler_gamma, scheduler_t0, scheduler_tmult, scheduler_eta_min):
    scheduler_name = str(scheduler_name).lower()
    if scheduler_name == 'multistep':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(scheduler_milestones),
            gamma=float(scheduler_gamma),
        )
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(scheduler_t0),
            T_mult=int(scheduler_tmult),
            eta_min=float(scheduler_eta_min),
        )
    if scheduler_name == 'step':
        first_milestone = list(scheduler_milestones)[0] if len(list(scheduler_milestones)) > 0 else 20
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(first_milestone),
            gamma=float(scheduler_gamma),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def apply_warmup_lr(optimizer, base_lrs, epoch, warmup_epochs, warmup_start_factor):
    if warmup_epochs <= 0:
        return [group['lr'] for group in optimizer.param_groups]
    if epoch >= warmup_epochs:
        return [group['lr'] for group in optimizer.param_groups]

    progress = float(epoch + 1) / float(warmup_epochs)
    scale = float(warmup_start_factor) + (1.0 - float(warmup_start_factor)) * progress
    for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
        param_group['lr'] = base_lr * scale
    return [group['lr'] for group in optimizer.param_groups]


def save_teacher_checkpoint(path, teacher, metric_head, epoch, best_eer=None):
    payload = {
        'epoch': int(epoch),
        'backbone_state': teacher.state_dict(),
        'metric_head_state': metric_head.state_dict(),
    }
    if best_eer is not None:
        payload['best_eer'] = float(best_eer)
    torch.save(payload, path)


def load_backbone_init_checkpoint(model, ckpt_path, device):
    if not ckpt_path:
        return False, [], []
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'backbone_state' in state:
        state = state['backbone_state']
    elif isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if isinstance(state, dict) and any(k.startswith('backbone.') for k in state.keys()):
        state = {k.replace('backbone.', '', 1): v for k, v in state.items() if k.startswith('backbone.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return True, missing, unexpected


def train_teachers(
    source_loader, source_val_loader,
    num_teachers, epochs, lr, writer, model_save_path, src_train_num_cls,
    scheduler_name, scheduler_milestones, scheduler_gamma,
    scheduler_t0, scheduler_tmult, scheduler_eta_min,
    weight_decay, early_stop_patience, early_stop_min_delta, save_epoch,
    optimize_metric_head, metric_head_lr, use_amp,
    warmup_epochs, warmup_start_factor,
    metric_scale, metric_margin, focal_gamma, grad_clip_norm,
    init_ckpt
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(model_save_path, exist_ok=True)
    
    # 初始化教师模型和度量头
    teacher = ResNet_double(BasicBlock, [2,2,2,2]).to(device)
    if init_ckpt:
        loaded, missing, unexpected = load_backbone_init_checkpoint(teacher, init_ckpt, device)
        if loaded:
            logging.info(
                f"teacher_ init checkpoint loaded from {init_ckpt}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            print(
                f"teacher_ init checkpoint loaded from {init_ckpt}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
    arcface_loss = ArcFaceLoss(
        embedding_size=128,
        class_num=src_train_num_cls,
        s=metric_scale,
        m=metric_margin,
        gamma=focal_gamma,
    ).to(device)
    # arcface_loss = AdaFaceLoss(embedding_size=128, class_num=src_train_num_cls, s=64, m=0.6, gamma=2).to(device)    # TODO 注意这里DA的也要改

    optim_params = [{'params': teacher.parameters(), 'lr': lr}]
    if optimize_metric_head:
        optim_params.append({'params': arcface_loss.parameters(), 'lr': metric_head_lr})
    optimizer = optim.RMSprop(optim_params, lr=lr, weight_decay=float(weight_decay))
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        scheduler_milestones=scheduler_milestones,
        scheduler_gamma=scheduler_gamma,
        scheduler_t0=scheduler_t0,
        scheduler_tmult=scheduler_tmult,
        scheduler_eta_min=scheduler_eta_min,
    )
    # mmd_loss = MMDLoss(kernel_type='linear', kernel_mul=2.0, kernel_num=5, fix_sigma=None).to(device)

    # 初始化混合精度训练器（每个教师独立使用一个scaler，避免相互干扰）
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_eer = float('inf') 
    best_epoch = -1
    no_improve = 0
    for epoch in tqdm(range(epochs)):
        current_lrs = apply_warmup_lr(
            optimizer,
            base_lrs,
            epoch,
            warmup_epochs,
            warmup_start_factor,
        )
       
        teacher.train()
        epoch_loss = 0.0

        src_iter = iter(source_loader)
        steps = len(source_loader)

        for _ in range(steps):
            optimizer.zero_grad()
            # 安全获取源域和目标域batch
            try:
                s_imgs, s_labels = next(src_iter)
            except StopIteration:
                src_iter = iter(source_loader)
                s_imgs, s_labels = next(src_iter)

            s_imgs, s_labels = s_imgs.to(device), s_labels.to(device)

            # 混合精度前向传播（核心修改）
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):  # 启用半精度计算
                s_convs, s_feats = teacher(s_imgs)
                # 损失计算在半精度上下文内
                loss_sup = arcface_loss(s_feats, s_labels)
                loss = loss_sup 
                # loss_ada = mmd_loss(s_feats, s_feats)
                # loss = loss_sup + loss_ada

            # 混合精度反向传播（核心修改）
            scaler.scale(loss).backward()  # 缩放损失，避免梯度下溢
            if float(grad_clip_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), float(grad_clip_norm))
                if optimize_metric_head:
                    torch.nn.utils.clip_grad_norm_(arcface_loss.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)         # 根据缩放梯度更新参数
            scaler.update()                # 更新缩放器状态

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps
        writer.add_scalar(f'teacher_/train_loss', avg_loss, epoch)
        writer.add_scalar(f'teacher_/lr', current_lrs[0], epoch)
        if optimize_metric_head and len(current_lrs) > 1:
            writer.add_scalar(f'teacher_/metric_head_lr', current_lrs[1], epoch)
        lr_msg = f'lr={current_lrs[0]:.6f}'
        if optimize_metric_head and len(current_lrs) > 1:
            lr_msg += f', metric_head_lr={current_lrs[1]:.6f}'
        logging.info(f'teacher_ Epoch {epoch} train_loss={avg_loss:.6f}, {lr_msg}')
        print(f'teacher_ Epoch {epoch} train_loss={avg_loss:.6f}, {lr_msg}')
        metrics = validate(teacher, source_val_loader) # 验证（验证阶段无需混合精度，保持默认精度）

        if epoch + 1 >= int(warmup_epochs):
            scheduler.step()  # warmup 结束后再进入主调度器

        write_info(avg_loss, metrics, writer, epoch, prefix=f'teacher_source')
        if metrics['eer'] < best_eer - float(early_stop_min_delta):
            best_eer = metrics['eer']
            best_epoch = epoch
            no_improve = 0
            save_teacher_checkpoint(
                os.path.join(model_save_path, 'teacher_best.pth'),
                teacher,
                arcface_loss,
                epoch,
                best_eer=best_eer,
            )
            logging.info(f'teacher_ best updated: epoch={epoch}, best_eer={best_eer:.4f}')
            print(f'teacher_ best updated: epoch={epoch}, best_eer={best_eer:.4f}')
        else:
            no_improve += 1
        if epoch % int(save_epoch) == 0:
            save_teacher_checkpoint(
                os.path.join(model_save_path, f'teacher_epoch{epoch}.pth'),
                teacher,
                arcface_loss,
                epoch,
                best_eer=best_eer,
            )

        if int(early_stop_patience) > 0 and no_improve >= int(early_stop_patience):
            logging.info(
                f'teacher_ early stop at epoch={epoch}, best_epoch={best_epoch}, '
                f'best_eer={best_eer:.4f}, patience={early_stop_patience}'
            )
            print(
                f'teacher_ early stop at epoch={epoch}, best_epoch={best_epoch}, '
                f'best_eer={best_eer:.4f}, patience={early_stop_patience}'
            )
            break

    return teacher


def recursive_namespace(data):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = recursive_namespace(v)
        return argparse.Namespace(**data)
    elif isinstance(data, list):
        return [recursive_namespace(d) for d in data]
    else:
        return data
    
def get_dataset(path, mean, std, split):
    """获取数据集"""
    cache_file = f"./data/cache/{path.replace('/', '_')}_{split}_cache.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            full_set, number_class = pickle.load(f)
    else:
        # 原始逻辑不变，获取全量的 Palmdata 数据集
        img_list = []
        if isinstance(path, str):
            path = [path]
        for p in path:
            img_list += list_pictures(p)
        # 假设 get_all_image 返回 data_paths & labels
        max_numbers = 200 #此文件夹下最大的类别数
        data_paths, labels = get_all_image(img_list, 1, max_numbers, 0)
        number_class = max(labels) + 1

        # 构造 dataset
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((124,124)),
                transforms.RandomCrop(112),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((124,124)),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        full_set = Palmdata(path, transform, data_paths, labels, split=split)
        with open(cache_file, 'wb') as f:
            pickle.dump((full_set, number_class), f)
    return full_set, number_class

def get_dataloader(path, mean, std, batch_size, num_workers, shuffle, split):
    """返回 (dataloader, num_classes)"""
    try:  # 添加异常捕获
        full_set, number_class = get_dataset(path, mean, std, split)
        dataloader = DataLoader(
            full_set, 
            batch_size=batch_size,
            shuffle=shuffle, 
            num_workers=num_workers,
            drop_last=(split == 'train')
        )
        return dataloader, number_class
    except Exception as e:
        logging.error(f"数据加载失败（路径：{path}）：{str(e)}")
        raise  # 抛出错误便于调试

def build_args():
    """构建命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_configs/train_one.yaml', help='配置文件路径')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # 解析命令行参数
    args = build_args()
    opts = yaml.safe_load(open(args.config, 'r'))
    opts = recursive_namespace(opts)
    save_path = opts.save_path
    print(opts)
    os.makedirs(save_path, exist_ok=True)
    shutil.copy(args.config, os.path.join(save_path, os.path.basename(args.config)))
    
    logging.basicConfig(
        filename=f'{save_path}/training.log',
        level=logging.INFO,
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    writer = SummaryWriter(log_dir=os.path.join(save_path, 'tensorboard'))
    source_path = opts.source_path[0]
    batch_size = opts.batch_size
    num_workers = opts.num_workers
    epochs = opts.epochs
    lr = float(opts.lr)
    scheduler_name = getattr(opts, 'scheduler', 'multistep')
    scheduler_milestones = getattr(opts, 'scheduler_milestones', [10, 40, 80, 120, 160])
    scheduler_gamma = float(getattr(opts, 'scheduler_gamma', 0.5))
    scheduler_t0 = int(getattr(opts, 'scheduler_t0', 10))
    scheduler_tmult = int(getattr(opts, 'scheduler_tmult', 2))
    scheduler_eta_min = float(getattr(opts, 'scheduler_eta_min', 1e-6))
    weight_decay = float(getattr(opts, 'weight_decay', 1e-4))
    early_stop_patience = int(getattr(opts, 'early_stop_patience', 25))
    early_stop_min_delta = float(getattr(opts, 'early_stop_min_delta', 1e-4))
    save_epoch = int(getattr(opts, 'save_epoch', 20))
    optimize_metric_head = bool(getattr(opts, 'optimize_metric_head', False))
    metric_head_lr = float(getattr(opts, 'metric_head_lr', lr))
    use_amp = bool(getattr(opts, 'use_amp', True))
    warmup_epochs = int(getattr(opts, 'warmup_epochs', 0))
    warmup_start_factor = float(getattr(opts, 'warmup_start_factor', 0.2))
    metric_scale = float(getattr(opts, 'metric_scale', 64.0))
    metric_margin = float(getattr(opts, 'metric_margin', 0.5))
    focal_gamma = float(getattr(opts, 'focal_gamma', 0.0))
    grad_clip_norm = float(getattr(opts, 'grad_clip_norm', 0.0))
    init_ckpt = str(getattr(opts, 'init_ckpt', '') or '')


    # 1) 加载源域 train/val
    src_train_loader,  src_train_num_cls = get_dataloader(
        source_path+"/cent_train", settings.DATA_TRAIN_MEAN, settings.DATA_TRAIN_STD,
        batch_size, num_workers, shuffle=True, split='train'
    )
    src_val_loader, src_val_num_cls  = get_dataloader(
        source_path+"/cent_test", settings.DATA_TRAIN_MEAN, settings.DATA_TRAIN_STD,
        batch_size, num_workers, shuffle=False, split='test'
    )
    print("实际训练类别数：", src_train_num_cls)  # 若不等于150，则必须修改
    
    logging.info("Source domain train: {} images, {} classes".format(len(src_train_loader.dataset), src_train_num_cls))
    logging.info("Source domain val: {} images, {} classes".format(len(src_val_loader.dataset), src_val_num_cls))

    # 查看 DataLoader 的部分属性
    print("Batch size:", src_train_loader.batch_size)
    print("Number of workers:", src_train_loader.num_workers)
    print("Optimize metric head:", optimize_metric_head)
    print("Metric head lr:", metric_head_lr)
    print("Use AMP:", use_amp)
    print("Warmup epochs:", warmup_epochs)
    print("Warmup start factor:", warmup_start_factor)
    print("Metric scale:", metric_scale)
    print("Metric margin:", metric_margin)
    print("Focal gamma:", focal_gamma)
    print("Grad clip norm:", grad_clip_norm)
    print("Init ckpt:", init_ckpt if init_ckpt else "None")
    # print("All attributes and methods of DataLoader:", dir(src_train_loader))
    # 遍历 DataLoader，查看第一个批次的数据维度
    for batch_data, batch_labels in src_train_loader:
        print("批次数据维度（Data shape）:", batch_data.shape)  # 输出: torch.Size([16, 3, 224, 224])
        print("批次标签维度（Label shape）:", batch_labels.shape, "labels内容：", batch_labels)  # 输出: torch.Size([16])
        break  # 只查看第一个批次，避免重复输出

    logging.info("Start training teachers")
    print("Start training teachers")
    teachers = train_teachers(
        src_train_loader,
        src_val_loader,
        1,
        epochs, lr, writer,
        os.path.join(save_path, 'model/adapt/teachers'),
        src_train_num_cls,
        scheduler_name,
        scheduler_milestones,
        scheduler_gamma,
        scheduler_t0,
        scheduler_tmult,
        scheduler_eta_min,
        weight_decay,
        early_stop_patience,
        early_stop_min_delta,
        save_epoch,
        optimize_metric_head,
        metric_head_lr,
        use_amp,
        warmup_epochs,
        warmup_start_factor,
        metric_scale,
        metric_margin,
        focal_gamma,
        grad_clip_norm,
        init_ckpt,
    )
    print("Finished training teachers")
    logging.info("Finished training teachers")
