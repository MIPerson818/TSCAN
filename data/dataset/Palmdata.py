import torch.utils.data as Data
from torchvision.datasets.folder import default_loader
import os
from os.path import join as ospj
from tqdm import tqdm

class Palmdata(Data.Dataset):
    """
    Palm dataset for person re-identification.
    """
    def __init__(self, data_path, transform, data_, labels_, split='train'):
        if not split in ['train', 'test']:
            raise Exception('Invalid dataset split.')
        self.transform = transform
        self.loader = default_loader
        self.split = split
        self.data_path = data_path
        self.data_ = data_
        self.labels_ = labels_

        if split == 'train':
            self.imgs, self.labels = self.data_, self.labels_ #self.get_all_image(self.data_path, 1, self.class_number, self.label_1)
        elif split == 'test':
            self.imgs, self.labels = self.data_, self.labels_ #self.get_all_image(self.data_path, 0, self.class_number, self.label_1)
                   
    @staticmethod
    def get_all_image(self, path, flag=1, class_number=0, label_1=0 ):
        
        train_datas = []
        train_lables = []
        roi_path = path
        i = label_1
        class_list = []
        img_num = []
        for image_list in tqdm(roi_path):  
            
            class_name = '_'.join(image_list.split('/')[:-1])
            if class_name not in class_list:#如果该类还没有被加入，第一次加入该类
                class_list.append(class_name)
                i = i + 1
                image_label = i-1
                img_num.append(0)
            else:
                image_label = class_list.index(class_name) + label_1
            #     train_lables.append(int(i-1))
            if image_label < class_number: #86个人，选取训练集
                img_num[image_label-label_1] = img_num[image_label-label_1] + 1
                if flag == 1:
                    # if image_label < 200: #86个人，选取训练集
                    #     if img_num[image_label] < 6:
                            train_lables.append(image_label)
                            train_datas.append(image_list)
                            
                if flag == 0:
                    # if image_label >= 200 : #86个人，选取训练集
                    #     if img_num[image_label] >= 6:
                            train_lables.append(image_label)
                            train_datas.append(image_list)
                            # img_num[image_label] = img_num[image_label] + 1
        
        print('====== image number',len(train_datas))
        print('====== label number',len(train_lables))
                        
        return train_datas, train_lables


    def __getitem__(self, index):
        path = self.imgs[index]
        label = self.labels[index]

        img = self.loader(path)
        
        if self.transform is not None:
            img = self.transform(img)
        # print(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    
    

    # @staticmethod
    # def id(file_path):
    #     """
    #     file_path: unix style file path
    #     return: person id
    #     """
        
    #     #XJTU-UP
    #     path = file_path.split('/')[-2]
    #     if path.split('_')[0] == 'L':
    #         id_ = int(path.split('_')[1])
    #     elif path.split('_')[0] == 'R':
    #         id_ = int(path.split('_')[1])+100
        
    #     '''
    #     #MPD
    #     path = file_path.split('/')[-2]
    #     if path.split('_')[-1] == 'L':
    #         id_ = int(path.split('_')[-2])
    #     elif path.split('_')[-1] == 'R':
    #         id_ = int(path.split('_')[-2])+200
    #     '''
    #     '''
    #     #tongji PolyU
    #     path = file_path.split('/')[-2]        
    #     id_ = int(path)  
    #     '''
    #     '''
    #     #CASIA
    #     path = file_path.split('/')[-2]
    #     if path.split('_')[1] == 'L':
    #         id_ = int(path.split('_')[0])
    #     elif path.split('_')[1] == 'R':
    #         id_ = int(path.split('_')[0])+312

    #     '''     
    #     '''   
    #     #IIT-D
    #     path = file_path.split('/')[-2]
    #     if path.split('_')[0] == 'Left':
    #         id_ = int(path.split('_')[1])
    #     elif path.split('_')[0] == 'Right':
    #         id_ = int(path.split('_')[1])+230
    #     '''
    #     return id_
    #     '''
    #     return int(file_path.split('/')[-1].split('_')[0])
    #     '''


    # @staticmethod
    # def camera(file_path):
    #     """
    #     file_path: unix style file path
    #     return: camera id
    #     """
    #     return int(file_path.split('/')[-1].split('_')[1][1])

    # @property
    # def ids(self):
    #     """
    #     return: person id list corresponding to dataset image paths
    #     """
    #     return [self.id(path) for path in self.imgs]

    # @property
    # def unique_ids(self):
    #     """
    #     return: unique person ids in ascending order
    #     """
        
    #     return sorted(set(self.ids))

    # @property
    # def cameras(self):
    #     """
    #     return: camera id list corresponding to dataset image paths
    #     """
    #     return [self.camera(path) for path in self.imgs]
