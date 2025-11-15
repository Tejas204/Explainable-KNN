# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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
        self.explanations.update({query: query_label})

        for key in self.neighbours:
            explanation, label = train_dataset[self.neighbours[key]]
            self.explanations.update({explanation: label})

        fig, axs = plt.subplots(2, 3, figsize=(8, 4))
        axs = axs.flatten()
        for i, (explanation, label) in enumerate(self.explanations.items()):
            axs[i].imshow(explanation.permute(1, 2, 0))
            if i == 0:
                axs[i].set_title("Query: " + str(label))
            else:
                axs[i].set_title(label)
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
        # For all images in the train dataset, compute distance, store as distance: image index
        for index in range(len(train_dataset)):
            image, _ = train_dataset[index]
            self.distances.update({self.compute_euclidean_distance(image, query): index})

        
        # Sort the dictionary in ascending
        self.distances = dict(sorted(self.distances.items(), key=lambda item:item[0]))

        # Compute the top k shorted distances and the indices of those images
        self.neighbours = {key: value for i, (key, value) 
                           in enumerate(self.distances.items()) if i < k}
        

        # Compute labels of top-k labels
        neighbour_indices = list(list(self.neighbours.values()))
        top_k_labels = []
        for i in neighbour_indices:
            top_k_labels.append(train_dataset[i][1])

        # Conduct majority vote and give label to the query
        top_k_labels_frequency = Counter(top_k_labels)
        majority_label = top_k_labels_frequency.most_common(1)
        query_label = majority_label[0][0]

        # Display explanations
        self.display_explanation(query, query_label)


# Testing
model = KNN()
model.compute_knn(query=test_dataset[2][0])