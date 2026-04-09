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
from DA.DA import train_teachers, train_student
from DA.train_dann import train_teachers_dann, train_student_dann
from configs import settings # 包含 DATA_TRAIN_MEAN, DATA_TRAIN_STD
import argparse
import yaml

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
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((124,124)),
            transforms.RandomCrop(112),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 颜色抖动
            transforms.ToTensor(),
            # AddGaussianNoise(mean=0., std=5),  # 添加高斯噪声
            # lambda tensor: add_noise_with_snr(tensor, 5),  # 添加5dB信噪比的白噪声   
            transforms.Normalize(mean, std)
        ])  # 你在 settings 中定义过的

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
            drop_last=True  # 补充：避免最后一个批次尺寸不足导致的问题
        )
        return dataloader, number_class
    except Exception as e:
        logging.error(f"数据加载失败（路径：{path}）：{str(e)}")
        raise  # 抛出错误便于调试

def build_args():
    """构建命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_configs/DA_distill_V0.0.4.yaml', help='配置文件路径')
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
    target_paths = opts.target_paths
    batch_size = opts.batch_size
    num_workers = opts.num_workers
    epochs = opts.epochs
    lr = float(opts.lr)

    # 合并所有验证数据加载器
    all_val_datasets = []

    # 1) 加载源域 train/val
    src_train_loader,  src_train_num_cls = get_dataloader(
        source_path+"/cent_train", settings.DATA_TRAIN_MEAN, settings.DATA_TRAIN_STD,
        batch_size, num_workers, shuffle=True, split='train'
    )
    src_val_loader, src_val_num_cls  = get_dataloader(
        source_path+"/cent_test", settings.DATA_TRAIN_MEAN, settings.DATA_TRAIN_STD,
        batch_size, num_workers, shuffle=False, split='test'
    )
    all_val_datasets.append(src_val_loader.dataset)
    
    logging.info("Source domain train: {} images, {} classes".format(len(src_train_loader.dataset), src_train_num_cls))
    logging.info("Source domain val: {} images, {} classes".format(len(src_val_loader.dataset), src_val_num_cls))
    print("Source domain train: {} images, {} classes".format(len(src_train_loader.dataset), src_train_num_cls))
    print("Source domain val: {} images, {} classes".format(len(src_val_loader.dataset), src_val_num_cls))

    # 查看 DataLoader 的部分属性
    print("Batch size:", src_train_loader.batch_size)
    print("Number of workers:", src_train_loader.num_workers)
    # print("All attributes and methods of DataLoader:", dir(src_train_loader))
    # 遍历 DataLoader，查看第一个批次的数据维度
    for batch_data, batch_labels in src_train_loader:
        print("批次数据维度（Data shape）:", batch_data.shape)  # 输出: torch.Size([16, 3, 224, 224])
        print("批次标签维度（Label shape）:", batch_labels.shape, "labels内容：", batch_labels)  # 输出: torch.Size([16])
        break  # 只查看第一个批次，避免重复输出

    # 2) 加载每个目标域 train/val
    tgt_train_loaders = []
    tgt_val_loaders = []
    for tp in target_paths:
        t_tr, t_tr_num_cls = get_dataloader(
            tp+"/cent_train", settings.DATA_TRAIN_MEAN, settings.DATA_TRAIN_STD,
            batch_size, num_workers, shuffle=True, split='train')
        t_val, t_val_num_cls = get_dataloader(
            tp+"/cent_test", settings.DATA_TRAIN_MEAN, settings.DATA_TRAIN_STD,
            batch_size, num_workers, shuffle=False, split='test')
        tgt_train_loaders.append(t_tr)
        tgt_val_loaders.append(t_val)
        all_val_datasets.append(t_val.dataset)
    logging.info("Target 0 domains train: {} images, {} classes".format(
        len(tgt_train_loaders[0].dataset), t_tr_num_cls))
    logging.info("Target 0 domains val: {} images, {} classes".format(
        len(tgt_val_loaders[0].dataset), t_val_num_cls))
    
    # 查看dataloader中图像路径是否正确 
    for i in range(len(tgt_train_loaders)):
        for batch_data, batch_labels in tgt_train_loaders[i]:
            print("Target {} 批次数据维度（Data shape）:".format(i), batch_data.shape)  # 输出: torch.Size([4, 3, 112, 112])
            print("Target {} 批次标签维度（Label shape）:".format(i), batch_labels.shape)  # 输出: torch.Size([4])
            break  # 只查看第一个批次，避免重复输出
    
    # 2) 合并所有验证数据集
    combined_val_dataset = ConcatDataset(all_val_datasets)
    combined_val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)

    logging.info("Combined val: {} images, {} classes".format(len(combined_val_loader.dataset), src_train_num_cls))
    print("Combined val: {} images, {} classes".format(len(combined_val_loader.dataset), src_train_num_cls))


    # 3) 训练教师
    if os.path.exists(os.path.join(save_path,'model/adapt/teachers/teacher_0.pth')) or opts.train_teachers == False:
        from models.resnet import ResNet_double, BasicBlock
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载训练好的模型
        teachers_path = [os.path.join(save_path, f'model/adapt/teachers/teacher_{i}.pth') for i in range(len(target_paths)) ]
        teachers = []
        for i in range(len(teachers_path)):
            t_path = teachers_path[i]
            assert os.path.exists(t_path), "Teacher {} not found".format(t_path)
            # 创建模型实例
            teacher_model = ResNet_double(BasicBlock, [2, 2, 2, 2]).to(device)
            # 加载模型参数
            teacher_model.load_state_dict(torch.load(t_path))
            teachers.append(teacher_model)
            logging.info("Loaded trained teacher_{} from {}".format(i,t_path))
            print("Loaded trained teacher_{} from {}".format(i,t_path))
    elif opts.train_teachers == True:
        logging.info("Start training teachers")
        print("Start training teachers")

        teachers = train_teachers_dann(
        # teachers = train_teachers(
            src_train_loader,
            src_val_loader,
            tgt_train_loaders,
            tgt_val_loaders,
            len(target_paths),
            epochs, lr, writer,
            os.path.join(save_path, 'model/adapt/teachers')
        )
        print("Finished training teachers")
        logging.info("Finished training teachers")
    else:
        raise ValueError("No trained teachers found and train_teachers is False")

    # 4) 训练学生
    if opts.train_student == True:
        logging.info("Start training student")
        print("Start training student")
        student = train_student_dann(
        # student = train_student(
            teachers,
            src_train_loader,
            tgt_train_loaders,
            combined_val_loader,
            epochs * 2, lr, writer,  os.path.join(save_path, 'model/adapt/student'),
            a=opts.distill_alpha, b=opts.distill_beta
        )
        print("Finished training student")
        logging.info("Finished training student")
    writer.close()