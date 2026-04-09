import torch.utils.data as Data
from torchvision.datasets.folder import default_loader
import os
from os.path import join as ospj
import random
import numpy as np

class TongJi_dataset(Data.Dataset):
    """
    TongJi palmprint dataset with open-set protocol (single domain).
    """
    def __init__(self, data_path=None, transform=None, split='train', 
                 train_ratio=0.8, test_with_labels=True):
        if not split in ['train', 'test']:
            raise Exception('Invalid dataset split. Must be "train" or "test".')
            
        self.transform = transform
        self.loader = default_loader
        self.split = split
        self.data_path = data_path if data_path is not None else '/home/repository/PalmDatas/tongji'  # 单domain路径
        self.train_ratio = train_ratio
        self.seed = 42
        self.test_with_labels = test_with_labels

        # 加载数据并获取映射字典
        self.imgs, self.labels, self.class_to_idx, self.idx_to_class = self.get_all_image()
        
        # 开集识别属性
        self.is_open_set = True
        self.known_classes = list(self.class_to_idx.keys()) if split == 'train' else []
        self.num_known_classes = len(self.known_classes)

    def get_all_image(self):
        # 固定随机种子确保可复现
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # 1. 收集所有类别（单domain下的人员文件夹）
        class_list = []
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")
        
        for person_folder in os.listdir(self.data_path):
            person_path = ospj(self.data_path, person_folder)
            if os.path.isdir(person_path):
                class_list.append(person_folder)  # 类别名称直接使用人员文件夹名（单domain无需额外前缀）
        
        # 2. 划分训练/测试类别
        num_classes = len(class_list)
        num_train_classes = int(num_classes * self.train_ratio)
        train_classes = random.sample(class_list, num_train_classes)
        test_classes = [c for c in class_list if c not in train_classes]
        used_classes = train_classes if self.split == 'train' else test_classes
        
        # 3. 创建类别-索引映射
        class_to_idx = {}
        idx_to_class = {}
        for label_idx, class_name in enumerate(used_classes):
            class_to_idx[class_name] = label_idx
            idx_to_class[label_idx] = class_name
        
        # 4. 收集当前split的图像和标签
        images = []
        labels = []
        for class_name in used_classes:
            person_path = ospj(self.data_path, class_name)
            # 获取该类别下所有图像
            image_files = [
                ospj(person_path, f) for f in os.listdir(person_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ]
            # 收集数据
            images.extend(image_files)
            labels.extend([class_to_idx[class_name]] * len(image_files))
        
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
        """统计每个类别（按索引）的图像数量"""
        class_count = {}
        for label in self.labels:
            class_count[label] = class_count.get(label, 0) + 1
        return class_count

    def get_num_classes(self):
        """返回当前split的类别数量"""
        return len(self.class_to_idx)


def main():
    import torch
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    # 配置
    data_path = '/home/repository/PalmDatas/tongji'
    transform = transforms.Compose([
        transforms.Resize((124, 124)),
        transforms.RandomCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ratio = 0.8

    # 创建数据集
    train_dataset = TongJi_dataset(
        data_path=data_path,
        transform=transform,
        split='train',
        train_ratio=train_ratio
    )
    test_dataset = TongJi_dataset(
        data_path=data_path,
        transform=transform,
        split='test',
        train_ratio=train_ratio
    )

    # 数据加载器
    batch_size = 24
    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 基本信息
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    print(f"训练类别数: {train_dataset.get_num_classes()}, 测试类别数: {test_dataset.get_num_classes()}")

    # 查看批次数据
    for batch_data, batch_labels in train_loader:
        print("\n训练集批次数据维度:", batch_data.shape)
        print("训练集批次标签维度:", batch_labels.shape)
        break

    for batch_data, batch_labels in test_loader:
        print("\n测试集批次数据维度:", batch_data.shape)
        print("测试集批次标签维度:", batch_labels.shape)
        print("测试集批次标签示例:", batch_labels)
        break

    # 统计类别图像数量（带类别名称）
    train_class_count = train_dataset.count_images_per_class()
    test_class_count = test_dataset.count_images_per_class()

    print("\n训练集中每个类别的图像数量：")
    for label_idx, count in sorted(train_class_count.items())[:1]:
        class_name = train_dataset.idx_to_class[label_idx]
        print(f"类别索引 {label_idx}: {count} 张图像, 类别名称: {class_name}")

    print("\n测试集中每个类别的图像数量：")
    for label_idx, count in sorted(test_class_count.items())[:1]:
        class_name = test_dataset.idx_to_class[label_idx]
        print(f"类别索引 {label_idx}: {count} 张图像, 类别名称: {class_name}")


if __name__ == "__main__":
    main()