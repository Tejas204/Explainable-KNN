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
k = 11
batch_size = 1

# Data loaders and datasets
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Model
class KNN():
    def __init__(self, k, batch_size):
        super(KNN, self).__init__()
        self.neighbours = k
        self.batch_size = batch_size
        self.neighbours = {}
        self.explanations = {}
        self.distances = {}
        self.class_dictionary = {0: "Airplane", 1: "Automobile", 2: "Bird", 3: "Cat", 4: "Deer", 5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9: "Truck"}

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

        fig, axs = plt.subplots(2, 6, figsize=(8, 4))
        axs = axs.flatten()
        for i, (explanation, label) in enumerate(self.explanations.items()):
            axs[i].imshow(explanation.permute(1, 2, 0))
            if i == 0:
                axs[i].set_title("Predicted Label:\n" + self.class_dictionary[label])
            else:
                axs[i].set_title(self.class_dictionary[label])
        plt.show()
        
    # ---------------------------------------------------------------------------------------------------------
    # @Function: compute_euclidean_distance
    # @Args: image, query
    # @Returns: euclidean distance (float)
    # @Desc: Compute euclidean distance between the query and the image
    # ---------------------------------------------------------------------------------------------------------
    def compute_euclidean_distance(self, image, query):
        image = image.numpy()
        query = query.numpy()
        euclidean_distance = np.sum(np.sum(np.square(image - query)))
        return euclidean_distance
    
    # ---------------------------------------------------------------------------------------------------------
    # @Function: compute_error_rate
    # @Args: predicted_labels, original_labels (array-like)
    # @Returns: error_rate (float)
    # @Desc: Compute euclidean distance between the query and the image
    # ---------------------------------------------------------------------------------------------------------
    def compute_error_rate(self, predicted_labels, original_labels):
        error_count = 0
        n_samples = len(predicted_labels)
        if len(predicted_labels) == len(original_labels):
            for i in range(len(predicted_labels)):
                if predicted_labels[i] != original_labels[i]:
                    error_count += 1
                else:
                    continue

        accuracy = error_count/n_samples
        error_rate = 1 - accuracy
        return error_rate

    # ---------------------------------------------------------------------------------------------------------
    # @Function: compute_knn
    # @Returns: query_label
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

        return query_label


# Testing


# Selecting optimal K
range_of_K = 10

for k_value in range(1, range_of_K+1):
    predicted_labels = []
    error_rates = []
    model = KNN(k=k_value, batch_size=batch_size)
    for i in range(len(test_dataset)):
        prediction = model.compute_knn(query=test_dataset[i][0])
        predicted_labels.append(prediction)

    error = model.compute_error_rate(predicted_labels, test_dataset[:][1])
    error_rates.append(error)

# Visualize
plt.plot([value+1 for value in range(range_of_K)], error_rates)
plt.xlabel("K-values")
plt.ylabel("Error rates")
plt.title("K-values vs Error rate")
plt.show()
