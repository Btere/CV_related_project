
from typing import List, Tuple
import numpy as np
from  sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.ticker as mticker
%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor


from dataset import load_dataset, load_dataset


def show_image(train_image)-> None:
    """We want to visualize the dataset downloaded"""
    col, row = 5, 5
    for i in range(1, col * row + 1):
        sample_idx = torch.randint(len(train_image), size=(1,)).item()
        img = train_image[sample_idx]
        figure = plt.figure(figsize=(9, 9))
        figure.add_subplot(row, col, i)
        plt.axis("off")
# Squeeze is used to remove the single-dimensional entries from the shape of an array.
        plt.imshow(img.squeeze(), cmap='gray')
    plt.show()

#show_images = show_image(train_images)

def show_transformed_images(dataset: int, num_images: int = 10)-> None:
    """Visualize transformed images"""
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.numpy().squeeze()  # Remove batch and channel dimensions
        mean = 0.5
        std = 0.5
        image = image * std + mean  # Denormalize
        plt.subplot(1, num_images, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap='gray')
        plt.xlabel(f'Label: {label}')
    plt.show()

# Display transformed images
#show_transformed_images(train_dataset)

def plot_confusion_matrix(y_true, y_pred):
  """We want to plot the confusion matrix to see the performance of the model"""
  cm = confusion_matrix(y_true, y_pred)
  return cm

#prediction, label = plot_confusion_matrix()

def plot_training_and_validation_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc) -> None:
    """plotting graphs to see how the model performs.

    Args:
        train_loss: Loss of the trained model
        train_acc: Correct prediction
        val_acc: validation set accuracy
    """
    epochs = np.arange(10)+1
    plt.subplot(111)
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Training and validation loss when batch size is 64')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
    plt.show()
    plt.subplot(111)
    plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Training and validation accuracy when batch size is 64')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    plt.legend()
    plt.grid('off')
    plt.show()

#return_varaible_name = plot_training_and_validation_loss_and_accuracy(train_loss, val_loss, train_acc, val_acc)