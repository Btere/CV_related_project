from typing import List, Tuple
import pathlib as Path
import dataclasses as dataclass

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
from torch.optim import Adam

from dataset import load_dataset, prepare_dataloaders
from visualization import plot_confusion_matrix

@dataclass
class ModelConfig:
    optimizer_type: str = "Adam"                #Default optimizer type
    learning_rate: float = 0.01 
    criterion: str = "CrossEntropyLoss"


class FashionMnistModel(nn.Module):
  def __init__(self, device: torch.Tensor) -> None:
    super().__init__()
    self.device = device
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 2) #
    self.bn1 = nn.BatchNorm2d(num_features=32)                          #we apply batchnorm, which must be same as the out_channel(extract the feature map)
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 2)
    self.bn2 = nn.BatchNorm2d(num_features=64)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
    self.bn3 = nn.BatchNorm2d(num_features=128)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=2)

    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
    self.fc2 = nn.Linear(in_features=512, out_features=256)
    self.fc3 = nn.Linear(in_features=256, out_features=10)                  #number of output class:out_feature

    self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the pipeline of the model.

        Args:
            x: Model input data.

        Returns:
            Output generated by the model.
        """
        x = self.conv1(x)
        print(f'After conv1: {x.shape}')
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        print(f'After pool1: {x.shape}')

        x = self.conv2(x)
        print(f'After conv1: {x.shape}')
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        print(f'After pool2: {x.shape}')

        x = self.conv3(x)
        print(f'After conv3: {x.shape}')
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        print(f'After pool3: {x.shape}')

        x = self.flatten(x)
        print(f'After flatten: {x.shape}')
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x
    

    def train_model(self, dataloader: DataLoader, epoch: int = 5) -> Tuple[float, float]:
        self.train()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = getattr(torch.optim, self.config.optimizer_type)(self.parameters(), lr=self.config.learning_rate)
        #loss_fn = nn.CrossEntropyLoss()                                 #3.define loss function, cos it is multi-class clssifcation problem
        #optimizer = Adam(self.parameters(), lr=0.001)                  # 4. Optimizer
        for epoch in range(epoch):
            running_loss: float = 0.0
            correct: int = 0
            total: int = 0
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                #forward pass
                prediction = self(images)
                print(prediction)
                #calculate loss
                loss = loss_fn(prediction, labels)
                print(loss)
                #backward pass, we update the weights of the loss function
                loss.backward()
                #update weights
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
                predicted = prediction.argmax(axis=1)                   # Converts the model's output probabilities (or logits) into class predictions.

                labels = torch.argmax(labels, dim=1)
                correct += (predicted == labels).sum().item()           ## Count correct predictions

                total += labels.size(0)                                     # Update the total number of samples
                average_loss = running_loss / len(dataloader)
                print(f'Epoch [{epoch + 1} / {self.epoch}], Loss: {average_loss:.4f}')

            accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')

        print("Finished Training")

        #return average_loss, accuracy

    def save_model(self, save_path: Path, model_name: str) -> None:
        """We want to save the trained model
        Args:
        model: Model to be saved.
        save_path: Path where the model will be saved.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / model_name)

    def load_model(self, model_path) -> nn.Module:
        """We want to load the trained model
        Args:
        model: Model to be loaded.
        save_path: Path where the model is saved.
        """
        model = torch.load(model_path)
        return model


    def validate_model(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model based on dataloader and specific criterion.

        Args:
            val_loader: Dataset loader to use for validation.
            criterion: Loss function used during validation.

        Returns:
            Current loss and accuracy.
        """
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                prediction = self(inputs)
                loss = criterion(prediction, labels)
                running_loss += loss.item()

                predicted = prediction.argmax(axis=1)
                print(predicted)
                labels = torch.argmax(labels, dim=1)

                correct += (predicted == labels).sum().item()
                print(correct)
                total += labels.size(0)
                final_loss = running_loss / len(val_loader)
                final_accuracy = 100 * correct / total

                return final_loss, final_accuracy
            
    def evaluate_model(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate the model based on dataloader and specific criterion.

        Args:
            val_loader: Dataset loader to use for validation.
            criterion: Loss function used during validation.

        Returns:
            Current loss and accuracy.
        """
        images = torch.tensor(images)
        labels = torch.tensor(labels)
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.eval()
        prediction = self.double()(inputs)
        loss = criterion(prediction, labels)

        predicted = prediction.argmax(axis=1)
        labels = torch.argmax(labels, dim=1)

        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        #cnf_matrix = confusion_matrix(labels.cpu(), predicted.cpu())
        #logging.disable(logging.DEBUG)  # disable not needed debug logging from matplotlib
        #plt.figure(figsize=(10, 10))
        #plot_confusion_matrix(y_pred, y_true, normalize=True,
                           # title='Confusion matrix, with normalization')
        #plt.show()
        #return loss, accuracy