import torch
import onnx
 
""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from utils import get_network
from tqdm import tqdm
import time


parser = argparse.ArgumentParser()
parser.add_argument('-num_classes', type=int, help='number of class', default=100)
parser.add_argument('-net', type=str, help='net type', default='mobilefacenet_base') #模型结构
parser.add_argument('-gpu', type=str, help='id of gpu device(s) to be used', default='0')
parser.add_argument('--cuda', action='store_true', default=True, help='whether to use cuda')
args = parser.parse_args()
model = get_network(args)

path = './model/HF/mobilefacenet_base-150-regular'
model_path = path + '.pth'

model.load_state_dict(torch.load(model_path))


im = torch.zeros((1, 3, 112, 112)).cuda()
for p in model.parameters():
    p.requires_grad = False
 
model.eval()
model.float()
# model = model.fuse()
f =  path + '.onnx'
# imput = torch.randn(1,1,128,128)
torch.onnx.export(model, 
                    im, f, opset_version=11, 
                    input_names=['img'], 
                    output_names=['out'])#, 
                    # dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                    #             'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                    #             })
 

model_onnx = onnx.load(f)

onnx.save(model_onnx, f)
