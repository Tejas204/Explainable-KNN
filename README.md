# Explainable-KNN

This repository contains the code for an explainable KNN model. The model explains why it assigned a label to an image based on the k-majority vote.

## Dependencies

1. Python
2. PyTorch
3. Numpy
4. Matplotlib

## Functionality

The KNN algorithm works as it is, however a few changes have been made. The top-k neighbours of the query image are stored in a data structure. Next, through a series of function calls, the top-k neighbours and their labels are displayed in a grid, with the help of `matplotlib`.

We also use the elbow method to choose the optimal value of k.
