import torch
import torch.nn as nn
from torch.nn import functional as F

import numbers

from losses.common import euclidean_dist, cosine_dist


class multi_DMMLLoss(nn.Module):
    """
    DMML loss with center support distance and hard mining distance.

    Args:
        num_support: the number of support samples per class.
        distance_mode: 'center_support' or 'hard_mining'.
    """
    def __init__(self, num_support, distance_mode='hard_mining', margin=0.4, gid=None):
        super().__init__()

        if not distance_mode in ['center_support', 'hard_mining']:
            raise Exception('Invalid distance mode for DMML loss.')
        if not isinstance(margin, numbers.Real):
            raise Exception('Invalid margin parameter for DMML loss.')

        self.num_support = num_support
        self.distance_mode = distance_mode
        #self.margin = margin
        self.gid = gid
        self.thresh = 0.5
        self.margin = 0.05

        self.scale_pos = 2
        self.scale_neg = 40

    def forward(self, feature, label):
        feature = F.normalize(feature)
        batch_size = feature.size(0)
        feature = feature.cpu()
        label = label.cpu()
        classes = torch.unique(label)  # torch.unique() is cpu-only in pytorch 0.4
        if self.gid is not None:
            feature, label, classes = feature.cuda(self.gid), label.cuda(self.gid), classes.cuda(self.gid)
        num_classes = len(classes)
        num_query = label.eq(classes[0]).sum() - self.num_support

        support_inds_list = list(map(
            lambda c: label.eq(c).nonzero()[:self.num_support].squeeze(1), classes))
        query_inds = torch.stack(list(map(
            lambda c: label.eq(c).nonzero()[self.num_support:], classes))).view(-1)
        query_samples = feature[query_inds]

        if self.distance_mode == 'center_support':
            center_points = torch.stack([torch.mean(feature[support_inds], dim=0)
                for support_inds in support_inds_list])
            dists = euclidean_dist(query_samples, center_points)
        elif self.distance_mode == 'hard_mining':

            dists = []
            max_self_dists = []
            loss=list()
            for i, support_inds in enumerate(support_inds_list):
                # dist_all = euclidean_dist(query_samples, feature[support_inds])
                dist_all = -cosine_dist(query_samples, feature[support_inds])
                pos_pair_ = dist_all[i*num_query:(i+1)*num_query]
                neg_pair_ = dist_all[0:i*num_query]
                n_dist = dist_all[(i+1)*num_query:]
                neg_pair_ = torch.cat([neg_pair_, n_dist], 0)

                pos_pair_ = pos_pair_.reshape(1, -1)
                pos_pair_ = torch.squeeze(pos_pair_)
                neg_pair_ = neg_pair_.reshape(1,-1)
                neg_pair_ = torch.squeeze(neg_pair_)
                
                neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
                pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

                # weighting step
                pos_loss = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
                neg_loss = 1.0 / self.scale_neg * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
                loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss)# / batch_size
        return loss
