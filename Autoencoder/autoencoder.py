from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from NeuralNetwork.neural_network import Layer
from NeuralNetwork.activations import ReLU, Sigmoid
from NeuralNetwork.convolutional import Conv_2D, Flatten
from NeuralNetwork.dropout import Dropout
from NeuralNetwork.pooling import MaxPool, Upsample, Concatenate, Reshape
import cupy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)
X = cp.asarray(X).reshape(-1, 8, 8) / 16.0
X = cp.expand_dims(X, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# MODEL

model_array = [
    Conv_2D(3, 8, 1),
    ReLU(),
    Conv_2D(3, 32, 8),
    ReLU(),
    MaxPool(2),
    Conv_2D(3, 64, 32),
    ReLU(),
    MaxPool(2),
    Flatten(),
    Layer(256, 512),
]

mu = Layer(512, 256, 'linear')
log_sigma = Layer(512, 256, 'linear')
concat = Concatenate()

model_array_second_pass = [
    Layer(256, 256),
    Layer(256, 512),
    Reshape((32, 4, 4)),
    Conv_2D(3, 64, 32),
    ReLU(),
    Upsample(2),
    Conv_2D(3, 32, 64),
    ReLU(),
    Conv_2D(3, 1, 32),
    Sigmoid()
]

def model_forward(array, array_, mu, sigma, x):
    out = x
    for layer in array:
        out = layer.forward(out)

    # print(out.shape)

    mu_out = mu.forward(out)
    log_sigma_out = sigma.forward(out)

    epsilon = cp.random.randn(*mu_out.shape)
    out = mu_out + cp.exp(log_sigma_out) * epsilon

    for layer in array_:
        out = layer.forward(out)

    return out, mu_out, log_sigma_out, epsilon

def model_backward(array, array_, mu, sigma, d_loss, d_m_kl, d_l_kl, lr, epsilon, l):
    out = d_loss
    for layer in reversed(array_):
        out = layer.backward(out, lr)
    
    out_i = out + d_m_kl
    out_j = (out * cp.exp(l) * epsilon) + d_l_kl

    mu_out = mu.backward(out_i, lr)
    log_sigma_out = sigma.backward(out_j, lr)

    out = mu_out + log_sigma_out

    for layer in reversed(array):
        out = layer.backward(out, lr)

    return out

# print(model_forward(model_array, model_array_second_pass, mu, log_sigma, X[0])[0].shape)
# print(model_backward(model_array, model_array_second_pass, mu, log_sigma, X[0], lr).shape)

# Plotting setup
plt.ion()

fig, (ax1, ax2) = plt.subplots(1,2)
im1 = ax1.imshow(cp.random.randn(8,8).get(), cmap='gray', vmin=0, vmax=1)
im2 = ax2.imshow(cp.random.randn(8,8).get(), cmap='gray', vmin=0, vmax=1)
plt.show(block=False)

# HYPERPARAMS
lr = 0.005
epochs = 25

for epoch in range(epochs):
    
    metrics = {
        "accuracy":0,
        "val_accuracy":0,
        "loss":0,
        "val_loss":0
    }

    # batch = cp.random.choice(len(X_train), size=256, replace=False)
    # batch = X_train[batch]
    cp.random.shuffle(X_train)
    for obs in tqdm(X_train):
        # TRAINING
        pred, m, l, epsilon = model_forward(model_array, model_array_second_pass, mu, log_sigma, obs)

        mse = cp.mean(cp.square(pred - obs))
        d_mse = 4 * (pred - obs) / pred.size 

        # KL Divergence
        kl_loss = -0.5 * cp.mean(1 + 2*l - cp.square(m) - cp.exp(2*l)) * 0.1
        d_m_kl = m / m.size * 0.1
        d_l_kl = (cp.exp(2*l) - 1) / l.size * 0.1

        metrics["loss"] += mse + kl_loss

        model_backward(model_array, model_array_second_pass, mu, log_sigma, d_mse, d_m_kl, d_l_kl, lr, epsilon, l)

    cp.random.shuffle(X_test)
    for v_obs in X_test:
        # VALIDATION
        pred, m, l, _ = model_forward(model_array, model_array_second_pass, mu, log_sigma, v_obs)

        mse = cp.mean(cp.square(pred - v_obs))

        # KL Divergence
        kl_loss = -0.5 * cp.mean(1 + 2*l - cp.square(m) - cp.exp(2*l))

        metrics["val_loss"] += mse + kl_loss

    if epoch % 1 == 0:
        im1.set_data(v_obs.squeeze(0).get())
        im2.set_data(pred.squeeze(0).get())
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001) 

    lr *= 0.99

    print(f"Epoch {epoch}: {metrics['loss'] / len(X_train):.4f} | V_Loss: {metrics['val_loss'] / len(X_test):.4f}")

from NeuralNetwork.saving import save_autoencoder



save_autoencoder("vae_digits.pkl", model_array, mu, log_sigma, model_array_second_pass)