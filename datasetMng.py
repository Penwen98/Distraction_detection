# Libraries
import sys
import numpy as np
from tqdm import tqdm
from time import time
from PIL import Image

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

def dataTransformer():
    global train_ds, test_ds, train_loader, test_loader, len_train, len_test
    
    tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Create Dataset
    TRAIN_ROOT = sys.path[0] + "/Revitsone-5classes/train"
    TEST_ROOT = sys.path[0] + "/Revitsone-5classes/test"
    
    train_ds = ImageFolder(TRAIN_ROOT, transform = tfm)
    test_ds = ImageFolder(TEST_ROOT, transform = tfm)
    
    # Length of Train and Test Datasets
    len_train = len(train_ds)
    len_test = len(test_ds)
    # Index Mapping
    train_ds.class_to_idx
    {'other_activities': 0,
     'safe_driving': 1,
     'talking_phone': 2,
     'texting_phone': 3,
     'turning': 4}
    # Data Loader
    train_loader = DataLoader(train_ds, batch_size = 30, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size = 30, shuffle=True)

    