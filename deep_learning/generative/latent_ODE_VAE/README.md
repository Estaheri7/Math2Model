# Latent ODE-VAE Implementation Using PyTorch

This notebook demonstrates the implementation of a **Latent ODE Variational Autoencoder (Latent ODE-VAE)** from scratch using PyTorch. A Latent ODE-VAE combines Variational Autoencoders (VAEs) with Neural ODEs to model time-evolving latent variables in a continuous-time framework. This is especially useful for modeling temporal sequences like video frames or irregularly sampled time series.

You can find orginal papers,  [Neural ODEs paper](https://arxiv.org/abs/1806.07366) and [Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907).

## Imports and Dataset Preparation

The dataset used is MNIST which is customized to RotatingMNIST (e.g., 5-frame sequences of MNIST digits rotating). Images are flattened, normalized, and reshaped into sequences of vectors over time.

---

## Model Components

### Encoder

The `Encoder` maps input sequences to a latent distribution $q(\mathbf{z}_0 \mid \mathbf{x}_{1:T})$, using stacked linear layers to predict the mean and log-variance of the latent variable $\mathbf{z}_0 \in \mathbb{R}^D$. Reparameterization is used to sample from this distribution.

### LatentODEfunc

Defines the neural ODE function

$$
\frac{d\mathbf{z}}{dt} = f(\mathbf{z}, t),
$$

a small MLP modeling the dynamics of the latent space. It takes the current latent state and time, and outputs the derivative $\dot{\mathbf{z}}$.

### LatentODEBlock

Wraps the ODE function and integrates it over time using `torchdiffeq.odeint` to obtain the full trajectory $\mathbf{z}_{1:T}$ from the initial condition $\mathbf{z}_0$.

### Decoder

Maps each latent vector $\mathbf{z}_t$ to the reconstructed observation $\hat{\mathbf{x}}_t$ using a simple MLP. It produces either a mean or parameters for a Gaussian likelihood.

### LatentODEVAE

The top-level model combines all the components:
- **Encode** sequence $\rightarrow \mathbf{z}_0$
- **ODE Integrate**: $\mathbf{z}_{1:T} = \texttt{odeint}(f, \mathbf{z}_0, t_{1:T})$
- **Decode**: $\mathbf{x}_{1:T} = \text{Decoder}(\mathbf{z}_{1:T})$

---

## Loss Function

The total loss is the standard VAE Evidence Lower Bound (ELBO):

$$
\mathcal{L} = \underbrace{\mathrm{KL}\left(q(\mathbf{z}_0 \mid \mathbf{x}_{1:T}) \,\|\, p(\mathbf{z}_0)\right)}_{\text{Regularization}} - \underbrace{\mathbb{E}_{q(\mathbf{z}_0)} \left[ \log p(\mathbf{x}_{1:T} \mid \mathbf{z}_{1:T}) \right]}_{\text{Reconstruction}}
$$

- The KL divergence encourages the posterior $q(\mathbf{z}_0)$ to be close to the prior $p(\mathbf{z}_0) = \mathcal{N}(0, I)$.
- The reconstruction term is computed using MSE or a log-likelihood loss.

---

## Training the Model

Training proceeds using standard gradient descent (e.g., Adam optimizer). Each batch includes:
- Sampling $\mathbf{z}_0$ via reparameterization
- ODE integration to get latent trajectories
- Decoding to reconstruct the input
- Computing the ELBO loss and updating parameters

