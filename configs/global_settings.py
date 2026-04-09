""" configurations for this project

author baiyu
"""
import os
from datetime import datetime


DATA_TRAIN_MEAN = [0.4648, 0.4648, 0.4648] # IR
DATA_TRAIN_STD = [0.0641, 0.0641, 0.0641]
# DATA_TRAIN_MEAN = [0.6064, 0.4929, 0.4999] # RGB
# DATA_TRAIN_STD = [0.0599, 0.0614, 0.0668]


#directory to save weights file
CHECKPOINT_PATH = 'model'

#total training epoches
EPOCH = 150
MILESTONES = [60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 40








