from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.utils import make_grid
from torch.optim import SGD, Adam


def load_dataset() -> None:
    """Download dataset"""
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root='./Data_dir', train=True, download=True, transform=transform)
    train_images = train_dataset.data
    target_class = train_dataset.targets


    test_data = torchvision.datasets.FashionMNIST(root='./Data_dir', train=True, download=True, transform=transform)
    val_images = test_data.data
    val_targets = test_data.targets
#load_dataset = load_dataset()

def prepare_dataloaders(train_image: torch.tensor, traintargets: torch.tensor, test_image: torch.tensor, test_targets:torch.tensor, batch_size: int = 64, shuffle: bool = True, num_workers=2) -> Tuple[DataLoader, DataLoader]:
  """Prepare dataset for training in batches
  Args:
    train_images: Data used as train input.
    train_targets: Data used as target label.
    test_images: Data used as test input.
    test_targets: Data used as test target.
    batch_size: Number of samples that will be propagated through the network
    shuffle: If true- randomly shuffles data
  """
  train_data = TensorDataset(train_image, traintargets)
  train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

  test_data = TensorDataset(test_image, test_targets) #we can use this as a validation set as well
  test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

  return train_dataloader, test_dataloader

#train_dataloader, test_dataloader = prepare_dataloaders(train_images,target_class, val_images, val_target)