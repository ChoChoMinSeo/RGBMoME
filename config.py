import torch
import random
import numpy as np
import os
import pandas as pd

config={
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'seed': 1337,
    'batch_size':64,
    'lr':1e-3,
    'epochs': 10000,
    'early_stopping':10000,
    'aux_output': False
}
print(config['device'])
SAVE_PATH = './logs'
save_paths={
    'model':SAVE_PATH+'/models/',
    'val':SAVE_PATH+'/vals/'
}
# 시드 고정 함수
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
fix_seed(config['seed'])