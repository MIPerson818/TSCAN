# train.py

""" train network using pytorch

"""
import os
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
import gc

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from loss import make_loss
from utils import float_or_string, normalize
from loss.AttentionConsistency import AttentionConsistency
from loss.MMD import MMDLoss
from loss.contrastive_loss import SupConLoss
from loss.contrastive import ContrastiveLoss
from loss.adv import AdversarialLoss


def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(data_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
        
        optimizer.zero_grad()
        if args.loss_type == 'softmax':
            logits, feat = net(images)
        else:
            feat = net(images)

        if args.loss_type == 'softmax':
            loss = loss_function(logits, labels)
        # elif args.loss_type == 'arcface':
        #     loss = loss_function(logits, labels)
        elif args.loss_type == 'softmax-triplet':
            loss = args.alpha * loss_function['softmax'](logits, labels) \
                    + (1 - args.alpha) * loss_function['triplet'](feat, labels)
        else:
            loss = loss_function(feat, labels)

        
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(data_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]

        if batch_index % 30 ==0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(data_training_loader.dataset)
            ))

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        # writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {}h {}m'.format(epoch, (finish - start) // 3600, ((finish - start) % 3600) // 60))

def init_seed(args, gids):
    """
    Set random seed for torch and numpy.
    """
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    if gids is not None:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # np.random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    datapath = ['/media/bell/public/shaohuikai/Database/roi_1_10(wanzheng)/HF']#, \
    #                 '/media/shaohuikai/public1/ALL_image/Open/IR','/media/shaohuikai/public1/ALL_image/Open/IR_Mirror']
    # datapath = ['/media/shaohuikai/public1/ALL_image/Open/IR','/media/shaohuikai/public1/ALL_image/Open/IR_Mirror']
    resume = False
    checkpoint_path = './model/HF'
    weights_path = './pre_trained_model.pth' # pre-trained model path
    parser.add_argument('--num_classes', type=int, help='number of class', default=100) # num_classes 代码可以自适应确定，不需要提前輸入
    parser.add_argument('--data_path', nargs='+', type=str, default=datapath)
    parser.add_argument('--net', type=str, help='net type', default='mobilefacenet_base') # 模型结构　mobilefacenet_base  resnet18
    parser.add_argument('--flag', type=int, default='1')
    parser.add_argument('--checkpoint_path', type=str, default='./model')
    parser.add_argument('--gpu', type=str, help='id of gpu device(s) to be used', default='1')
    parser.add_argument('--loss_type', type=str,
                        help='loss used in training, can be chosen from \'softmax\', \'multi_similarity\', \
                        \'triplet\', \'softmax-triplet\', cosface, circle, \'lifted\', contrastive, arcface, \'npair\' and \'dmml\',\'multi_dmml',
                        default='arcface') #從裏面選擇損失函數
    parser.add_argument('--margin', type=float_or_string, help='margin parameter', default=0.5)
    parser.add_argument('--alpha', type=float,
                        help='balance parameter of softmax loss and triplet loss, \
                        ranging from 0 to 1.0', default=0.5)
    parser.add_argument('--cuda', action='store_true', default=True, help='whether to use cuda')
    parser.add_argument('--b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    # parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--manual_seed',type=int, help='manual seed for initialization', default=7)
    args = parser.parse_args()
    """
    Initialization
    """
    print('Initializing...')
    if args.cuda:
        gpus = ''.join(args.gpu.split())
        gids = [int(gid) for gid in gpus.split(',')]
    else:
        gids = None
    init_seed(args, gids)
    gc.collect()
    torch.cuda.empty_cache()
    net = get_network(args)
    print(args.data_path)
    print(args.net)
    #data preprocessing:
    data_training_loader, args.num_classes = get_training_dataloader(
        args.data_path,
        settings.DATA_TRAIN_MEAN,
        settings.DATA_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    print(args.num_classes)
    # loss_function = nn.CrossEntropyLoss()
    loss_function = make_loss(args, gids)

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9, eps=1e-08)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(data_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    if resume:
        net.load_state_dict(torch.load(weights_path))
    start = time.time()
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)

        if not epoch % 150:#settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    # writer.close()
