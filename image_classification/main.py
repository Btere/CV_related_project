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

from dataset import load_dataset, prepare_dataloaders
from visualization import show_image,  plot_confusion_matrix, show_transformed_images
from model import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    train_images, target_class, val_images,  val_targets = load_dataset()
    show_images = show_image(train_images)
    show_transformed_images(train_dataset)  #train_dataset is in load_dataset, check it
    train_dataloader, test_dataloader = prepare_dataloaders(train_images,target_class, val_images, val_targets)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FashionMnistModel(device).to(device)

    plot_training_and_validation_loss_and_accuracy()



if __name__ == "__main__":
    main()