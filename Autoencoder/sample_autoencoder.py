import matplotlib.pyplot as plt
import cupy as cp
from NeuralNetwork.saving import load_model

def generate_images(decoder, num_samples=5, latent_dim=128):
    """Samples random noise and decodes it into novel images."""
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    
    for i in range(num_samples):
        # Draw from the standard normal prior p(z)
        z = cp.random.randn(latent_dim)
        
        # Pass the noise through the decoder
        out = z
        for layer in decoder:
            out = layer.forward(out)
            
        # The output is a CuPy array. Move it to the CPU for matplotlib
        # and reshape it to the 8x8 image dimension
        img = cp.asnumpy(out).reshape(8, 8)
        
        # Plotting
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Sample {i+1}")
        
    plt.suptitle("Generated Digits from Latent Space")
    plt.tight_layout()
    plt.show()

model = load_model("vae_digits.pkl")

generate_images(model["decoder"])