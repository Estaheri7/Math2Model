# Autoencoders: Mathematical Foundation and Applications

## What is an Autoencoder?

An **Autoencoder** is a type of neural network that learns to **compress** data into a lower-dimensional latent representation (encoding) and then **reconstruct** it back to the original form (decoding). It is an unsupervised learning model used primarily for **dimensionality reduction**, **denoising**, and **representation learning**.

---

## Architecture

An Autoencoder consists of two parts:

1. **Encoder**: Compresses the input into a latent space representation.
2. **Decoder**: Reconstructs the input from the latent representation.

Let:
- Input: $x \in \mathbb{R}^n$
- Encoding function: $z = f_{\theta}(x)$
- Decoding function: $\hat{x} = g_{\phi}(z)$

The full reconstruction function is:

$$
\hat{x} = g_{\phi}(f_{\theta}(x))
$$

The goal is to minimize the **reconstruction loss**:

$$
\mathcal{L}(x, \hat{x}) = \| x - \hat{x} \|_2^2 = \| x - g_{\phi}(f_{\theta}(x)) \|_2^2
$$

---

## Mathematical Details

### 1. **Encoder**

Typically a series of linear or nonlinear transformations:

$$
z = f_{\theta}(x) = \sigma(W_e x + b_e)
$$

Where:
- $W_e \in \mathbb{R}^{k \times n}$, $b_e \in \mathbb{R}^k$
- $k \ll n$ (compression)

### 2. **Decoder**

Maps latent vector $z$ back to the input space:

$$
\hat{x} = g_{\phi}(z) = \sigma(W_d z + b_d)
$$

Where:
- $W_d \in \mathbb{R}^{n \times k}$, $b_d \in \mathbb{R}^n$

### 3. **Loss Function**

For reconstruction:

$$
\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \| x^{(i)} - \hat{x}^{(i)} \|^2
$$

Other variants may add regularization or noise (see below).

---

## Types of Autoencoders

| Type | Description |
|------|-------------|
| **Vanilla Autoencoder** | Basic encoder-decoder structure |
| **Denoising Autoencoder** | Learns to reconstruct from noisy input |
| **Sparse Autoencoder** | Adds sparsity penalty to latent space |
| **Variational Autoencoder (VAE)** | Learns a distribution in latent space (probabilistic) |
| **Convolutional Autoencoder** | Uses conv layers for image-based encoding/decoding |

---