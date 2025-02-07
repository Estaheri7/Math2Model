# Variational Autoencoder (VAE)

This notebook contains an implementation of a Variational Autoencoder (VAE) using PyTorch. The VAE is trained on the CIFAR-10 dataset to generate new images.

## Variational Autoencoder

A Variational Autoencoder (VAE) is a generative model that learns to encode data into a latent space and then decode it back to the original space. Unlike traditional autoencoders, VAEs impose a probabilistic structure on the latent space, which allows for the generation of new data points by sampling from the latent space.

### VAE Architecture

The VAE consists of two main components:
1. **Encoder**: Maps the input data to a latent space.
2. **Decoder**: Maps the latent space back to the original data space.

### Reparameterization Trick

The reparameterization trick is used to allow backpropagation through the stochastic sampling process. Instead of sampling directly from the distribution, we sample from a standard normal distribution and then shift and scale the samples using the mean and variance predicted by the encoder.

### Loss Function

The loss function for a VAE consists of two parts:
1. **Reconstruction Loss**: Measures how well the decoder reconstructs the input data.
2. **KL Divergence**: Measures how close the learned latent space distribution is to the prior distribution (usually a standard normal distribution).

The total loss is the sum of the reconstruction loss and the KL divergence.

```python
def vae_loss(x_recon, x, mu, log_var):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss
```

## Usage

### Training

To train the VAE, run the following code:

```python
vae = VAE(20, 3, 32).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

epochs = 15

for epoch in tqdm(range(epochs)):
    total_loss = 0
    for x, _ in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(x)
        loss = vae_loss(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader.dataset):.4f}")
```

### Generating Images

To generate new images using the trained VAE, run the following code:

```python
def generate_images(model, num_samples=10):
    with torch.no_grad():
        z = torch.randn(num_samples, 20)
        samples = model.decoder(z.to(device)).view(num_samples, 3, 32, 32)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 10))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
    plt.show()

generate_images(vae, 10)
```

This will display a set of generated images.