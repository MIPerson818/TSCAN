import os
import logging
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import pickle
import torch

from utils.utils import list_pictures, Palmdata, get_all_image  # 你的原 utils
from DA.DA import train_teachers, train_student
from configs import settings # 包含 DATA_TRAIN_MEAN, DATA_TRAIN_STD


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
    full_set, number_class = get_dataset(path, mean, std, split)
    dataloader = DataLoader(full_set, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
    return dataloader, number_class




if __name__ == "__main__":
    # 基本配置
    source_path = '/home/repository/PalmDatas/XJTUUP/xjtu/HUAWEI/Nature'
    target_paths = ['/home/repository/PalmDatas/XJTUUP/xjtu/iPhone/Nature',
                    '/home/repository/PalmDatas/XJTUUP/xjtu/LG/Nature',
                    '/home/repository/PalmDatas/XJTUUP/xjtu/MI8/Nature']
    writer = SummaryWriter()
    os.makedirs('model', exist_ok=True)
    batch_size = 24 #36
    num_workers = 8
    epochs = 200
    lr = 1e-4

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
    if os.path.exists('model/adapt'):
        from models.resnet import ResNet_double, BasicBlock
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载训练好的模型
        teachers_path = [f'model/adapt/teacher_{i}.pth' for i in range(len(target_paths)) ]
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
    else:
        logging.info("Start training teachers")
        print("Start training teachers")
        teachers = train_teachers(
            src_train_loader,
            src_val_loader,
            tgt_train_loaders,
            tgt_val_loaders,
            len(target_paths),
            epochs, lr, writer
        )
        print("Finished training teachers")
        logging.info("Finished training teachers")

    # 4) 训练学生
    logging.info("Start training student")
    print("Start training student")
    student = train_student(
        teachers,
        src_train_loader,
        tgt_train_loaders,
        combined_val_loader,
        epochs, lr, writer,
        a=1.0, b=0.5
    )
    print("Finished training student")
    logging.info("Finished training student")

    writer.close()