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
        self.neighbours = {}
        self.explanations = {}
        self.distances = {}

    # ---------------------------------------------------------------------------------------------------------
    # @Function: display_explanation
    # @Args: query, query_label
    # @Desc: Display the explanations in a 2*3 grid with labels
    # ---------------------------------------------------------------------------------------------------------
    def display_explanation(self, query, query_label):
        for key in self.neighbours:
            explanation, label = train_loader[self.neighbours[key]]
            self.explanations.update({label, explanation})

        fig, axs = plt.subplots(2, 3)
        axs = axs.flatten()
        for i in range(len(self.explanations)+1):
            if i == 0:
                axs[i].imshow(query.permute(1, 2, 0))
                axs[i].imshow(query.permute(1, 2, 0))
            axs[i].imshow(self.explanations[i].permute(1, 2, 0))
            axs[i].set_title(list(self.explanations.keys())[i])
        plt.show()
        
    # ---------------------------------------------------------------------------------------------------------
    # @Function: compute_euclidean_distance
    # @Args: image, query
    # @Desc: Compute euclidean distance between the query and the image
    # ---------------------------------------------------------------------------------------------------------
    def compute_euclidean_distance(self, image, query):
        image = image.numpy()
        query = query.numpy()
        euclidean_distance = np.sum(np.sum(np.square(image - query)))
        return euclidean_distance

    # ---------------------------------------------------------------------------------------------------------
    # @Function: compute_knn
    # @Args: query
    # @Desc: Compute euclidean distance, sort the top k neighbours and compute majority vote, display results
    # ---------------------------------------------------------------------------------------------------------
    def compute_knn(self, query):
        # For all images in the train loader, compute distance
        for index in range(len(train_loader)):
            image, label = train_loader[index]
            self.distances.update({self.compute_euclidean_distance(image, query): index})

        self.distances = dict(sorted(self.distances.items(), key=lambda item:item[0]))
        self.neighbours = {key: value for i, (key, value) 
                           in enumerate(self.distances.items()) if i < k}
        
        query_label = ""

        self.display_explanation(query, query_label)


# Testing
model = KNN()
model.compute_knn(query="Hey there")