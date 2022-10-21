import logging
import os
import random
import time
import math

import numpy as np
import scipy.spatial
import scipy.io as sio
import torch
import torch.nn.functional as F

def zero2eps(x):
    x[x == 0] = 1
    return x

def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum # row data sum = 1
    in_affnty = np.transpose(affinity/row_sum) # col data sum = 1 then transpose
    return in_affnty, out_affnty

def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False # False make training process too slow!
    torch.backends.cudnn.deterministic = True

def logger():
    '''
    '\033[0;34m%s\033[0m': blue
    :return:
    '''
    logger = logging.getLogger('PAGN')
    logger.setLevel(logging.DEBUG)

    if not os.path.exists('log/'):
        os.mkdir('log/')

    timeStr = time.strftime("[%m-%d]%H:%M:%S", time.localtime())

    txt_log = logging.FileHandler('log/'+ timeStr +'.log')
    txt_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', '%m/%d %H:%M:%S')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    # console + color
    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('\033[0;32m%s\033[0m' % '[%(asctime)s][%(levelname)s] %(message)s', '%m/%d %H:%M:%S')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger

def log_params(logger, config: dict):
    logger.info('--- Configs List---')
    for k in config.keys():
        logger.info('--- {:<18}:{}'.format(k, config[k]))

def GEN_S_GPU(label_1, label_2):
    aff = torch.matmul(label_1, label_2.T)
    affinity_matrix = aff.float()
    affinity_matrix = 1 / (1 + torch.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    return affinity_matrix

def int2bool(flag: int):
    '''

    :param flag: -1: False // 1: True
    :return:
    '''
    return False if flag == -1 else True

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


