# Coloring Autoencoder

This repository contains a Jupyter Notebook that demonstrates the implementation of a coloring autoencoder using PyTorch. The autoencoder is trained to colorize grayscale images from the CIFAR-10 dataset.

## Algorithm Explanation

The algorithm used in this notebook is based on an autoencoder architecture with residual blocks. The autoencoder consists of an encoder and a decoder:

1. **Encoder**: The encoder compresses the input grayscale image into a lower-dimensional representation. It uses convolutional layers followed by batch normalization and ReLU activation functions.

2. **Residual Blocks**: Residual blocks are used to improve the learning capability of the network by allowing gradients to flow through the network directly. This helps in training deeper networks.

3. **Decoder**: The decoder reconstructs the colorized image from the lower-dimensional representation. It uses transposed convolutional layers to upsample the feature maps back to the original image size.

## Files

- `COLORING_AE.ipynb`: The main Jupyter Notebook containing the implementation of the coloring autoencoder.
- `README.md`: This file, providing an overview of the project.

## Training

The training process involves the following steps:
1. Load the CIFAR-10 dataset.
2. Transform the images to grayscale.
3. Train the autoencoder to minimize the mean squared error (MSE) loss between the original and reconstructed images.
4. Evaluate the model on the test dataset.

## Results

The notebook includes a function to display the original, grayscale, and colorized images side by side for visual comparison.
