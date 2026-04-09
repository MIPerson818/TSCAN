import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaFaceLoss(nn.Module):
    """
    AdaFace Loss: Adaptive Face Recognition Loss with Dynamic Margin.
    Paper: https://arxiv.org/abs/2204.00964
    
    Args:
        embedding_size: 特征嵌入维度
        class_num: 类别数量
        s: 缩放因子
        m: 基础margin
        h: 动态margin的下限
        w: 动态margin的上限
        AdaFace 特有参数:
            l_a: 范数的下限
            u_a: 范数的上限
    """
    def __init__(self, embedding_size=128, class_num=100, s=64.0, m=0.6, 
                 h=0.8, w=0.4, l_a=2.0, u_a=25.0, gamma=2):
        super(AdaFaceLoss, self).__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.h = h  # 动态margin的下限
        self.l_a = l_a  # 范数的下限
        self.u_a = u_a  # 范数的上限
        self.gamma = gamma
        
        # 初始化权重
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # 预计算常量
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        # 交叉熵损失
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, label, img=None):
        """
        Args:
            input: 特征嵌入 [batch_size, embedding_size]
            label: 类别标签 [batch_size]
            img: 输入图像 [batch_size, channels, height, width] (用于计算图像范数)
        """
        # 特征归一化
        input_norm = torch.norm(input, p=2, dim=1, keepdim=True)  # [batch_size, 1]
        
        # 权重归一化
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 计算余弦相似度
        cosine = F.linear(F.normalize(input), weight_norm)  # [batch_size, class_num]
        
        # 计算图像质量 (使用特征范数作为代理)
        # 归一化范数到 [0, 1] 范围
        quality_score = (input_norm - self.l_a) / (self.u_a - self.l_a)
        quality_score = torch.clamp(quality_score, 0.0, 1.0)  # 确保在 [0,1] 范围内
        # print(quality_score)
        
        # 动态调整margin
        # 高质量图像使用较大的margin加强学习力度，低质量图像使用较小的margin
        margin = self.m * (self.h + (1.0 - self.h) * quality_score)

        # 计算sine
        # sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        
        # 计算 cos(theta + margin)
        phi = cosine * torch.cos(margin) - sine * torch.sin(margin)
        
        # 应用阈值，防止角度过大
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        # # 构建最终输出
        # one_hot = torch.zeros(cosine.size(), device=input.device)
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # # 动态调整缩放因子 (可选)
        # # 这里简化处理，保持原始缩放因子s
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # output *= self.s
        
        # # 应用focal loss
        # logp = self.ce(output, label)
        # p = torch.exp(-logp)
        # loss = (1 - p) ** self.gamma * logp
        
        # return loss.mean()
    

        # update y_i by phi in cosine
        output = cosine.clone()  # make backward works
        phi = phi.to(output.dtype)  # key step 转换类型一致
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        loss_input = output * self.s 
        logp = self.ce(loss_input, label)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()