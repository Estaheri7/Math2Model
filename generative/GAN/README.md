# Generative Adversarial Network (GAN) from Scratch

This repository contains an implementation of a Generative Adversarial Network (GAN) using PyTorch. The GAN is trained on the CIFAR-10 dataset to generate realistic images.

## Files

- `GAN.ipynb`: Jupyter notebook containing the implementation and training of the GAN.
- `README.md`: This file.

## Model Architecture

### Generator

The generator takes a random noise vector as input and generates an image. It consists of several transposed convolutional layers with batch normalization and ReLU activation functions.

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_channels=1, img_s=28):
        super().__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, img_s * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(img_s * 8),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(img_s * 8, img_s * 4, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(img_s * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(img_s * 4, img_s * 2, kernel_size=4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(img_s * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(img_s * 2, n_channels, kernel_size=2, stride=2, padding=2),
            nn.Tanh()
        )

    def forward(self, z: Tensor):
        return self.generator(z)
```

### Discriminator

The discriminator takes an image as input and outputs a probability indicating whether the image is real or fake. It consists of several convolutional layers with batch normalization and LeakyReLU activation functions.

```python
class Discriminator(nn.Module):
    def __init__(self, n_channels=1, img_s=28):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(n_channels, img_s * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(img_s * 2, img_s * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(img_s * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(img_s * 4, img_s * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(img_s * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(img_s * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        return self.discriminator(x)
```

## Training

The GAN is trained using the Binary Cross Entropy Loss function. The training process involves alternating between training the discriminator and the generator. The discriminator is trained to distinguish between real and fake images, while the generator is trained to produce images that are classified as real by the discriminator.

## Results

The notebook visualizes the training images, the loss curves for the generator and discriminator, and an animation of the generated images over the training epochs.

## Saving and Loading Models

The trained generator and discriminator models are saved as `generator.pth` and `discriminator.pth` respectively.
