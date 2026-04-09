from xml import dom
import torch.utils.data as Data
from torchvision.datasets.folder import default_loader
import os
from os.path import join as ospj
from tqdm import tqdm
import random
import numpy as np

class PolyU_dataset(Data.Dataset):
    """
    PolyU palmprint dataset for person re-identification with open-set protocol.
    """
    def __init__(self, data_path=None, transform=None, split='train', 
                 train_ratio=0.8, domain='Blue', test_with_labels=True):
        if not split in ['train', 'test']:
            raise Exception('Invalid dataset split.')
        if not domain in ['Blue', 'Green', 'NIR', 'Red']:
            raise Exception('Invalid domain. Must be one of: Blue, Green, NIR, Red')
            
        self.transform = transform
        self.loader = default_loader
        self.split = split
        self.data_path = data_path if data_path is not None else '/home/repository/PalmDatas/PolyU'
        self.domain = domain 
        self.train_ratio = train_ratio
        self.test_with_labels = test_with_labels
        self.seed = 42

        self.imgs, self.labels, self.class_to_idx, self.idx_to_class = self.get_all_image(
            self.data_path, self.domain, self.split, self.train_ratio, self.seed, self.test_with_labels
        )
        
        # 为开集识别准备的属性
        self.is_open_set = True
        self.known_classes = list(self.class_to_idx.values()) if split == 'train' else []
        self.num_known_classes = len(self.known_classes) if split == 'train' else 0

    def get_all_image(self, path, domain, split, train_ratio, seed, test_with_labels):
        # 设置随机种子确保结果可复现
        random.seed(seed)
        np.random.seed(seed)
        
        domain_path = ospj(path, domain)
        if not os.path.exists(domain_path):
            raise FileNotFoundError(f"Domain path not found: {domain_path}")
            
        # 收集所有类别（每个类别是一个文件夹）
        class_list = []
        for person_folder in sorted(os.listdir(domain_path)):
            person_path = ospj(domain_path, person_folder)
            if os.path.isdir(person_path):
                class_list.append(person_folder)
                
        # 划分训练集和测试集的类别
        num_classes = len(class_list)
        num_train_classes = int(num_classes * train_ratio)
        train_classes = random.sample(class_list, num_train_classes)
        test_classes = [c for c in class_list if c not in train_classes]
        used_classes = train_classes if split == 'train' else test_classes
        
        label_index = 0
        class_to_idx = {}
        idx_to_class = {}
        for class_name in used_classes:
            class_to_idx[class_name] = label_index
            idx_to_class[label_index] = class_name
            label_index += 1
            
        images = []
        labels = []
        for person_folder in used_classes:
            person_path = ospj(domain_path, person_folder)
            image_files = [ospj(person_path, f) for f in os.listdir(person_path) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            images.extend(image_files)
            labels.extend([class_to_idx[person_folder]] * len(image_files))
        
        return images, labels, class_to_idx, idx_to_class


    def __getitem__(self, index):
        path = self.imgs[index]
        label = self.labels[index]

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

    def count_images_per_class(self):
        class_count = {}
        for label in self.labels:
            if label not in class_count:
                class_count[label] = 1
            else:
                class_count[label] += 1
        return class_count

    def get_num_classes(self):
        """获取数据集中的类别总数（包括训练和测试类别）"""
        if self.split == 'train':
            return len(set(self.labels))
        else:
            if self.test_with_labels:
                return len(set(self.labels))
            else:
                return len(set(self.labels)) - 1  # 排除-1标签


def main():
    import torch
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    # 数据集路径
    data_path = '/home/repository/PalmDatas/PolyU'
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((124, 124)),
        transforms.RandomCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 训练集类别比例
    train_ratio = 0.8
    # 选择特定领域
    domain = 'Blue'

    # 创建训练集和测试集
    train_dataset = PolyU_dataset(data_path, transform, split='train', 
                                 train_ratio=train_ratio, domain=domain)
    test_dataset = PolyU_dataset(data_path, transform, split='test', 
                                train_ratio=train_ratio, domain=domain)

    # 创建数据加载器
    batch_size = 24
    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 打印训练集和测试集的大小
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 打印训练集和测试集的类别数量
    train_classes = set(train_dataset.labels)
    test_classes = set(test_dataset.labels)
    
    print(f"训练集中的已知类别数量: {len(train_classes)}")
    print(f"测试集中的未知类别数量: {len(test_classes)}")

    # 查看训练集第一个批次的数据
    for batch_data, batch_labels in train_loader:
        print("训练集批次数据维度（Data shape）:", batch_data.shape)
        print("训练集批次标签维度（Label shape）:", batch_labels.shape)
        break

    # 查看测试集第一个批次的数据
    for batch_data, batch_labels in test_loader:
        print("测试集批次数据维度（Data shape）:", batch_data.shape)
        print("测试集批次标签维度（Label shape）:", batch_labels.shape)
        print("测试集批次标签示例:", batch_labels)
        break

    # 统计训练集和测试集中每个类别的图像数量
    train_class_count = train_dataset.count_images_per_class()
    test_class_count = test_dataset.count_images_per_class()

    print("\n训练集中每个类别的图像数量：")
    for class_label, count in sorted(train_class_count.items()):
        print(f"训练类别idx {class_label}: {count} 张图像, 类别名称: {train_dataset.idx_to_class[class_label]}")
        break

    print("\n测试集中每个类别的图像数量：")
    for class_label, count in sorted(test_class_count.items()):
        print(f"测试类别idx {class_label}: {count} 张图像, 类别名称: {test_dataset.idx_to_class[class_label]}")
        break


if __name__ == "__main__":
    main()