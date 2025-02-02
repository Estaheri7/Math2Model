# Denoising Autoencoder (DAE)

This project demonstrates the implementation of a Denoising Autoencoder (DAE) using PyTorch. The DAE is trained to remove noise from images, specifically using the CIFAR-10 dataset.

## Algorithm Explanation

A Denoising Autoencoder is a type of neural network used to learn efficient codings of input data while removing noise. The network consists of two main parts:

1. **Encoder**: This part compresses the input image into a lower-dimensional representation.
2. **Decoder**: This part reconstructs the image from the compressed representation.

The network is trained to minimize the difference between the original and reconstructed images, effectively learning to remove noise from the input images.

## Files

- **denoising_AE.ipynb**: The main Jupyter notebook containing the implementation of the DAE.
- **README.md**: This file, providing an overview of the project.

## Usage

1. **Import Libraries**: The necessary libraries such as PyTorch, torchvision, and matplotlib are imported.
2. **Data Preparation**: The CIFAR-10 dataset is loaded and transformed.
3. **Noise Addition**: A function to add noise to the images is defined.
4. **Model Definition**: The DAE model is defined with an encoder and decoder.
5. **Training and Testing**: Functions to train and test the model are implemented.
6. **Visualization**: Functions to visualize the noisy and reconstructed images are provided.

## Training

The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer. The training process involves adding noise to the images, passing them through the DAE, and minimizing the reconstruction loss.

## Results

The training and testing losses are plotted to monitor the performance of the model. Additionally, the notebook includes functions to visualize the original, noisy, and reconstructed images.

## Conclusion

This project demonstrates the effectiveness of Denoising Autoencoders in removing noise from images. The implementation can be extended to other datasets and noise types for further experimentation.
