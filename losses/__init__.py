from torch.nn import CrossEntropyLoss
from losses.contrastive import ContrastiveLoss
from losses.triplet import TripletLoss
from losses.npair import NpairLoss
from losses.lifted import LiftedLoss
from losses.dmml import DMMLLoss
from losses.multi_similarity_loss import MultiSimilarityLoss
from losses.multi_dmml import multi_DMMLLoss
from losses.arcface import ArcFaceLoss
from losses.circle import SparseCircleLoss
from losses.cosface import CosFaceLoss
from losses.MMD import MMDLoss

def make_loss(args, gids):
    """
    Construct loss function(s).
    """
    gid = None if gids is None else gids[0]
    if args.loss_type == 'softmax':
        criterion = CrossEntropyLoss()
    elif args.loss_type == 'adaface':
        criterion = CrossEntropyLoss()
    elif args.loss_type == 'contrastive':
        criterion = ContrastiveLoss(margin=args.margin)
    elif args.loss_type == 'triplet':
        criterion = TripletLoss(margin=args.margin)
    elif args.loss_type == 'softmax-triplet':
        criterion = {'softmax': CrossEntropyLoss(),
                     'triplet': TripletLoss(margin=args.margin)}
    elif args.loss_type == 'npair':
        criterion = NpairLoss(reg_lambda=0.002, gid=gid)
    elif args.loss_type == 'lifted':
        criterion = LiftedLoss(margin=args.margin, gid=gid)
    elif args.loss_type == 'dmml':
        criterion = DMMLLoss(num_support=args.num_support, distance_mode=args.distance_mode,
                             margin=args.margin, gid=gid)   
    elif args.loss_type == 'multi_dmml':
        criterion = multi_DMMLLoss(num_support=args.num_support, distance_mode=args.distance_mode,
                             margin=args.margin, gid=gid)                               
    elif args.loss_type == 'multi_similarity':
        criterion = MultiSimilarityLoss(scale_pos=2, scale_neg=1)
    elif args.loss_type == 'arcface':
        criterion = ArcFaceLoss(embedding_size=128, class_num=args.num_classes, s=64, m=args.margin)
        # criterion = CrossEntropyLoss()
    elif args.loss_type == 'cosface':
        criterion = CosFaceLoss(embedding_size=512, class_num=args.num_classes, s=64, m=args.margin)
        # criterion = CrossEntropyLoss()
    elif args.loss_type == 'circle':
        criterion = SparseCircleLoss(m=args.margin, emdsize=128, class_num=args.num_classes, gamma=64)
    else:
        raise NotImplementedError

    return criterion
