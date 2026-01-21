import torch
import numpy as np
import random
import csv
import os
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_str=None):
    if device_str: return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CSVLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.log_dir / "metrics.csv"
        self.columns = ["epoch", "val_acc", "test_acc", "sparsity", "params"]
        
        if not self.file_path.exists():
            with open(self.file_path, mode='w', newline='') as f:
                csv.writer(f).writerow(self.columns)
            
    def log(self, metrics):
        with open(self.file_path, mode='a', newline='') as f:
            row = [metrics.get(col, "") for col in self.columns]
            csv.writer(f).writerow(row)