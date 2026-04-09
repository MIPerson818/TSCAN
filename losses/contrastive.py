import torch
import torch.nn as nn

import numbers

from losses.common import euclidean_dist
from losses.common import get_mask

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute similarity matrix
        inputs = nn.functional.normalize(inputs,p=2, dim=1, eps=1e-12)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        loss = list()
        c = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets==targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets!=targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)
            
            neg_loss = 0

            pos_loss = torch.sum(-pos_pair_+1) 
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
            loss.append(pos_loss + neg_loss)

        loss = sum(loss)/n
        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return loss#, prec, mean_pos_sim, mean_neg_sim


# class ContrastiveLoss(nn.Module):
#     """
#     Batch hard contrastive loss.
#     """
#     def __init__(self, margin):
#         super(ContrastiveLoss, self).__init__()
#         if not isinstance(margin, numbers.Real):
#             raise Exception('Invalid margin parameter for contrastive loss.')
#         self.margin = margin

#     def forward(self, feature, label):
#         # print(label)
#         feature = nn.functional.normalize(feature,p=2, dim=1, eps=1e-12)
#         distance = euclidean_dist(feature, feature, squared=True)
#         # print(distance)

#         positive_mask = get_mask(label, 'positive')
#         hardest_positive = (distance * positive_mask.float()).max(dim=1)[0]
#         p_loss = hardest_positive.mean()
        
#         negative_mask = get_mask(label, 'negative')
#         max_distance = distance.max(dim=1)[0]
#         not_negative_mask = ~negative_mask
#         negative_distance = distance + max_distance * (not_negative_mask.float())
#         hardest_negative = negative_distance.min(dim=1)[0]
#         n_loss = (self.margin - hardest_negative).clamp(min=0).mean()

#         con_loss = p_loss + n_loss
        
#         return con_loss
