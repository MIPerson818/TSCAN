#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
import numpy as np
from matplotlib import pyplot as plt
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader
from tqdm import tqdm
import time

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_classes', type=int, help='number of class', default=100)
    parser.add_argument('-net', type=str, help='net type', default='mobilefacenet_base')#模型结构
    parser.add_argument('-gpu', type=str, help='id of gpu device(s) to be used', default='0')
    parser.add_argument('--cuda', action='store_true', default=True, help='whether to use cuda')
    args = parser.parse_args()
    net = get_network(args)


    source_path = ['/media/bell/public/shaohuikai/Database/roi_1_10(wanzheng)/HF']
    model_path = './model/HF/mobilefacenet_base-150-regular.pth' 


    data_source_loader = get_test_dataloader(
        source_path,
        settings.DATA_TRAIN_MEAN,
        settings.DATA_TRAIN_STD,
        num_workers=4,
        batch_size=64,
        class_number=100, # 用多少个类别进行测试
        num_per=10, #每个类用多少张图进行测试
    )
    feature_dim = 128
    
    net.load_state_dict(torch.load(model_path))
    distance = 'consine' # Euclidean  consine
    net.eval()
    t0 = int(time.time())
    result_code = []
    source = torch.zeros(1,feature_dim).cuda()

    true_list = []
    false_list = []

    s_label=torch.tensor([],dtype=torch.long)

    with torch.no_grad():
        for n_iter, (image, label) in tqdm(enumerate(data_source_loader)):
            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                s_label = s_label.cuda()
                feat = net(image)
            # feat = feat.data.cpu()
            # feat = feat.numpy()
            source = torch.cat((source,feat))
            s_label = torch.cat((s_label,label))
        
    torch.cuda.empty_cache()
    source = source.data.cpu()
    source = source.numpy()
    source = source[1:,:] 
    
    result_code=np.reshape(source,[-1,feature_dim]) 
    print('--------',result_code.shape) 
    result_code /= np.maximum(1e-5, np.linalg.norm(result_code, axis=1, keepdims=True))          
    #-------------------------------------------------------------------------------------------------
    # used to obtain the DET curve and EER
    image_list = []
    for i in tqdm(range(len(result_code))): #匹配
        #print(i)
        for j in range(i+1, len(result_code)):
            if s_label[i] == s_label[j]:
                if distance == 'consine':
                    true_list.append(np.dot(result_code[i],result_code[j]))
                else:
                    true_list.append(np.sqrt(np.sum(np.square(result_code[i] - result_code[j]))))
            else:
                if distance == 'consine':
                    false_list.append(np.dot(result_code[i],result_code[j]))
                else:
                    false_list.append(np.sqrt(np.sum(np.square(result_code[i] - result_code[j]))))
    #-------------------------------------------------------------------------------------------------
    # used to get the threshold as different FAR
    print(max(label)+1)
    if distance == 'consine':
        false_list = sorted(false_list,reverse = True) #降序
    else:
        false_list = sorted(false_list,reverse = False) #升序
    #true_list = sorted(true_list) # 升序        
    total_true = len(true_list)
    total_false = len(false_list)
    FAR = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    m = 0
    for far in FAR:
        if far * total_false < 1:
            break
        thr = false_list[int(far * total_false)]
        for num, true_ in enumerate(true_list):
            if distance == 'consine':
                if true_ < thr:
                    m = m +1
            else:
                if true_ > thr:
                    m = m +1
                # if far == 0.0001:
                #     print('bad case', image_list[num], true_)
            #else:
            #    break
        TAR = (total_true - m) / total_true
        m = 0
        print('@FAR={}, Thr={}, TAR={}'.format(far,thr,TAR))
        
    t = int(time.time())
    print('Time elapsed: {}h {}m'.format((t - t0) // 3600, ((t - t0) % 3600) // 60))

    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))


if __name__ == '__main__':
    main()