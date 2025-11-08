# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
k = 5
batch_size = 1

# Data load
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Visualization
example = iter(train_loader)
samples, labels = next(example)
print(samples.shape, labels.shape)


# fig, axs = plt.subplots(2, 2)
# axs = axs.flatten()
# for i in range(batch_size):
#     axs[i].imshow(samples[i].permute(1, 2, 0))
# plt.show()


# Model
class KNN():
    def __init__(self):
        super(KNN, self).__init__()
        self.neighbours = k
        self.batch_size = batch_size
        self.neighbour_array = {}

    def display_explanation():
        pass

    def compute_knn(self, query):
        # For all images in the train loader
        # Compute distance

        # Add top k to the dictionary

        # Display as a grid
        print(f"Query: {query}")


# Testing
model = KNN()
model.compute_knn(query="Hey there")