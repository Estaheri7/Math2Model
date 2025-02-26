# Diffusion Models

Diffusion models are a class of generative models that learn to generate data by reversing a diffusion process. The diffusion process gradually adds noise to the data, and the model learns to reverse this process to generate new data samples.

## How Diffusion Models Work

1. **Forward Diffusion Process**: This process involves gradually adding Gaussian noise to the data over a series of time steps. The data becomes increasingly noisy until it is nearly indistinguishable from pure noise.

2. **Reverse Diffusion Process**: The model learns to reverse the forward diffusion process. Starting from pure noise, the model iteratively denoises the data to generate new samples that resemble the original data distribution.

### Mathematical Formulation

Let \( x_0 \) be the original data sample, and \( x_t \) be the noisy version of the data at time step \( t \). The forward diffusion process can be described as:

\[ x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon \]

where \( \epsilon \sim \mathcal{N}(0, I) \) is Gaussian noise, and \( \alpha_t \) is a noise schedule that controls the amount of noise added at each time step.

The reverse diffusion process aims to estimate the noise \( \epsilon \) added at each step and denoise the data iteratively. The model \( \epsilon_\theta(x_t, t) \) is trained to predict the noise at each time step.

The loss function used to train the model is typically the mean squared error (MSE) between the predicted noise and the actual noise:

\[ \mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right] \]

## Implementation in `diffusion.ipynb`

The provided Jupyter notebook `diffusion.ipynb` implements a simple diffusion model using the MNIST dataset. Below are the key components of the implementation:

1. **Data Loading**: The MNIST dataset is loaded and transformed into tensors.

2. **Model Architecture**: A simple U-Net architecture is defined to predict the noise at each time step.

3. **Training Loop**: The model is trained to minimize the MSE loss between the predicted noise and the actual noise.

4. **Image Generation**: After training, the model is used to generate new images by reversing the diffusion process.