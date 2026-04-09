""" helper function
"""
import os
import sys
import re
import datetime
from tqdm import tqdm
import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
sys.path.insert(0,("/home/workstation/Palm/PR_20241226_copy"))
from data.dataset.Palmdata import Palmdata

def float_or_string(args):
    """
    Tries to convert the string to float, otherwise returns the string.
    """
    try:
        return float(args)
    except (ValueError, TypeError):
        return args

def get_network(args):
    """ return given network
    """
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_class = args.num_classes)
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'mobilefacenet_base':
        from models.MobileFaceNet import mobilefacenet
        net = mobilefacenet()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes = args.num_classes)
    elif args.net == 'IR18':
        from models.IR import IR_18
        net = IR_18(input_size=(112,112))
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

class AddGaussianNoise:  
    def __init__(self, mean=0., std=1.):  
        self.mean = mean  
        self.std = std  
  
    def __call__(self, tensor):  
        """  
        Args:  
            tensor (Tensor): Tensor image of shape (C, H, W) to be added noise.  
  
        Returns:  
            Tensor: Noised tensor image.  
        """  
        # 确保tensor在[0, 1]范围内  
        assert tensor.max() <= 1 and tensor.min() >= 0  
          
        # 确保噪声在[0, 1]范围内  
        noise = torch.randn(tensor.size()) * self.std + self.mean  
        noise = noise.clamp(0, 1)  
          
        # 确保噪声的均值接近0，标准差接近指定的std  
        noise = noise - noise.mean()  
        noise = noise / noise.std() * self.std  
          
        # 将噪声添加到图像上  
        tensor = tensor + noise  
          
        # 确保图像值在[0, 1]范围内  
        tensor = tensor.clamp(0, 1)  
          
        return tensor 

def add_noise_with_snr(tensor, snr_db):  
    # 假设tensor的形状是[C, H, W]，并且值在[0, 1]之间  
    # 计算信号的均方值（这里假设为所有像素的平均功率）  
    signal_power = torch.mean(tensor ** 2)  
    # 根据SNR计算噪声的方差  
    noise_variance = signal_power / (10 ** (snr_db / 10))  
    # 生成与tensor相同形状的白噪声（标准正态分布）  
    noise = torch.randn(tensor.size()) * torch.sqrt((noise_variance.clone().detach()))  
    # 将噪声添加到tensor上，并确保值在[0, 1]之间  
    noisy_tensor = torch.clamp(tensor + noise, 0, 1)  
    return noisy_tensor  

def list_pictures( path):
        image_list = []
        imgType_list = ['jpg','bmp','png','jpeg','rgb','tif','JPG']
        for root1,dirs_name,files_name in os.walk(path):
            for i in files_name:
                file_name = os.path.join(root1, i)
                if file_name.split('.')[-1] in imgType_list:
                    image_list.append(file_name)
        return image_list

def get_all_image(path, flag=1, class_number=0, label_1=0 ):
        
        train_datas = []
        train_lables = []
        roi_path = path
        i = label_1
        class_list = []
        img_num = []
        for image_list in (roi_path):  

            class_name = '_'.join(image_list.split('/')[:-1]) # 得到样本的类别
            if class_name not in class_list:#如果该类还没有被加入，第一次加入该类
                class_list.append(class_name)
                i = i + 1
                image_label = i-1
                img_num.append(0)
            else:
                image_label = class_list.index(class_name) + label_1
            #     train_lables.append(int(i-1))
            if image_label < class_number: #　选取文件下多少个类做训练
                img_num[image_label-label_1] = img_num[image_label-label_1] + 1
                # if flag == 1:
                train_lables.append(image_label)
                train_datas.append(image_list)                            
        
        print('====== image number',len(train_datas))
        print('====== label number',len(train_lables))
                        
        return train_datas, train_lables

def get_training_dataloader(data_path, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        data_path: 测试数据路径        
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((124,124)),
        transforms.RandomCrop(112),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        # AddGaussianNoise(mean=0., std=5),  # 添加高斯噪声
        # lambda tensor: add_noise_with_snr(tensor, 5),  # 添加5dB信噪比的白噪声   
        transforms.Normalize(mean, std)
    ])
    data_ = [] 
    labels_ = []
    img_path = []
    for path in data_path:
        # print(path)
        img_path += list_pictures(path)
        # if labels_ == []:
        #     class_number = 0
        # else:
        #     class_number = int(max(labels_)) + 1 #　类别数量
    max_numbers = 100 #此文件夹下最大的类别数
    data1, labels1 = get_all_image(img_path, 1, max_numbers, 0)
    data_ = data1
    labels_ = labels1
    print(min(labels_))

    number_class = max(labels_) + 1

    train_set = Palmdata(path, transform_train, data_, labels_, split='train')
    
    data_training_loader = DataLoader(
        train_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return data_training_loader, number_class

def get_test_image(path, class_number=85, num_per=50 ):
        
        train_datas = []
        train_lables = []
        roi_path = path
        label_1 = 0
        i = 0
        class_list = []
        img_num = []
        for image_list in (roi_path):  
            class_name = '_'.join(image_list.split('/')[:-1]) #/media/shaohuikai/public1/ALL_image/RGB/datasetNew_L001/1.jpg
            if class_name not in class_list:#如果该类还没有被加入，第一次加入该类
                class_list.append(class_name)
                i = i + 1
                image_label = i-1
                img_num.append(0)
            else:
                image_label = class_list.index(class_name) + label_1
            
            if image_label < class_number: 
                img_num[image_label-label_1] = img_num[image_label-label_1] + 1
                if img_num[image_label] < num_per: #每类取多少图进行测试
                    train_lables.append(image_label)
                    train_datas.append(image_list)
                    # img_num[image_label] = img_num[image_label] + 1

        print('====== image number',len(train_datas))
        print('====== label number',len(train_lables))
                        
        return train_datas, train_lables

def get_test_dataloader(data_path, mean, std, batch_size=16, num_workers=2, class_number=0, num_per=0, shuffle=False):
    """ return test dataloader
    Args:
        class_number: 用多少个类别进行测试
        num_per: 每个类用多少张图进行测试
        data_path: 测试数据路径
        
    """

    transform_test = transforms.Compose([
        # transforms.RandomCrop(112, padding=4)
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        # AddGaussianNoise(mean=0., std=10),  # 添加高斯噪声
        # lambda tensor: add_noise_with_snr(tensor, 5),  # 添加5dB信噪比的白噪声
        transforms.Normalize(mean, std)
    ])

    data_ = [] 
    labels_ = []
    img_path = []
    for path in data_path:
        img_path += list_pictures(path)
        print(len(img_path))

    data1, labels1 = get_test_image(img_path, class_number, num_per)
    data_ = data1
    labels_ = labels1
    print(max(labels_)+1)
    test_set = Palmdata(path, transform_test, data_, labels_, split='test')
    data_test_loader = DataLoader(
        test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return data_test_loader

def compute_mean_std(palm_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([palm_dataset[i][1][:, :, 0] for i in range(len(palm_dataset))])
    data_g = np.dstack([palm_dataset[i][1][:, :, 1] for i in range(len(palm_dataset))])
    data_b = np.dstack([palm_dataset[i][1][:, :, 2] for i in range(len(palm_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x

if __name__ == '__main__':
    data_path = ['/home/repository/PalmDatas/XJTUUP/xjtu/HUAWEI/Nature']
    transform_test = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
    ])

    data_ = [] 
    labels_ = []
    img_path = []
    for path in data_path:
        print(path)
        img_path += list_pictures(path)
        print(len(img_path))
        # if labels_ == []:
        #     class_number = 0
        # else:
        #     class_number = int(max(labels_)) + 1 #　类别数量
    data1, labels1 = get_test_image(img_path, 1000, 1000)
    data_ = data1
    labels_ = labels1
    img = Palmdata(path, transform_test, data_, labels_, split='test')
    data_test_loader = DataLoader(
        img, shuffle=False, num_workers=0, batch_size=1)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for batch_index, (images, labels) in tqdm(enumerate(data_test_loader)):
        for d in range(3):
            mean[d] += images[:,d,:,:].mean()
            std[d] += images[:,d,:,:].std()
    mean.div_(len(data_test_loader))
    std.div_(len(data_test_loader))
    print(mean.data.cpu(), std.data.cpu())