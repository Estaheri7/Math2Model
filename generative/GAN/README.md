# Generative Adversarial Network (GAN)

This repository contains an implementation of a Generative Adversarial Network (GAN) using PyTorch. The GAN is trained on the MNIST dataset to generate handwritten digits.

## How It Works

The GAN consists of two neural networks: a Generator and a Discriminator. The Generator creates fake images from random noise, while the Discriminator tries to distinguish between real and fake images. The two networks are trained simultaneously in a game-theoretic framework.

### Generator

The Generator takes a random noise vector as input and transforms it into an image through a series of transposed convolutional layers. The architecture is defined as follows:

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_channels=1, img_s=28):
        super().__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, img_s * 4, kernel_size=7, bias=False),
            nn.BatchNorm2d(img_s * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(img_s * 4, img_s * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(img_s * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(img_s * 2, n_channels, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, z: Tensor):
        return self.generator(z)
```

### Discriminator

The Discriminator takes an image as input and outputs a probability indicating whether the image is real or fake. It uses a series of convolutional layers to extract features from the image. The architecture is defined as follows:

```python
class Discriminator(nn.Module):
    def __init__(self, n_channels=1, img_s=28):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(n_channels, img_s * 2, kernel_size=2, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_s * 2, img_s * 4, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(img_s * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_s * 4, img_s * 8, kernel_size=2, stride=2, bias=2),
            nn.BatchNorm2d(img_s * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(img_s * 8, 1, kernel_size=3, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        return self.discriminator(x)
```

## Loss Functions

The loss functions for the Generator and Discriminator are based on binary cross-entropy loss. The Discriminator's loss is the sum of the losses for real and fake images, while the Generator's loss is based on how well it can fool the Discriminator.

### Discriminator Loss

```python
D_error_real: Tensor = loss_fn(D_output, labels)
D_error_fake: Tensor = loss_fn(D_output, labels)
D_error = D_error_real + D_error_fake
```

### Generator Loss

```python
G_error: Tensor = loss_fn(D_output, labels)
```

## References

- The original GAN paper: [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
- PyTorch documentation: [PyTorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#)
