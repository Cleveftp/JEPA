from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from NeuralNetwork.neural_network import Sequential, Layer
from NeuralNetwork.activations import ReLU, Sigmoid
from NeuralNetwork.convolutional import Conv_2D, Flatten
from NeuralNetwork.dropout import Dropout
from NeuralNetwork.pooling import MaxPool
import cupy as cp
from tqdm import tqdm

X, y = load_digits(return_X_y=True)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))
y = cp.asarray(y)
X = cp.asarray(X).reshape(-1, 8, 8) / 16.0
X = cp.expand_dims(X, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_test.shape, X_train.shape)

# HYPERPARAMS
lr = 0.01
epochs = 25

model = Sequential(lr)
model.add_layers([
    Conv_2D(3, 4, 1),
    ReLU(),
    MaxPool(2),
    Conv_2D(3, 8, 4),
    ReLU(),
    MaxPool(2),
    Flatten(),
    Layer(32, 16),
    Layer(16, 10, 'Sigmoid')
])

metrics_array = {
        "accuracy":[],
        "val_accuracy":[],
        "loss":[],
        "val_loss":[]
    }

for epoch in range(epochs):
    
    metrics = {
        "accuracy":0,
        "val_accuracy":0,
        "loss":0,
        "val_loss":0
    }

    for obs, target in zip(tqdm(X_train), y_train):
        # TRAINING
        pred = model.forward(obs)

        # loss
        mse = cp.mean(cp.square(pred - target))
        d_mse = 2 * (pred - target) / pred.size

        metrics["loss"] += mse

        # Calculate accuracy
        if cp.argmax(pred) == cp.argmax(target):
            metrics["accuracy"] += 1

        model.backward(d_mse)

    for v_obs, v_target in zip(X_test, y_test):
        # VALIDATION
        v_pred = model.forward(v_obs)
        v_loss = cp.mean(cp.square(v_pred - v_target))

        metrics["val_loss"] += v_loss

        if cp.argmax(v_pred) == cp.argmax(v_target):
            metrics["val_accuracy"] += 1

    print(f"Epoch {epoch}: {metrics['loss']:.4f} | Accuracy: {metrics['accuracy']/len(X_train):4f} | V_Loss: {metrics['val_loss']:4f} | V_Accuracy: {metrics['val_accuracy']/len(X_test):4f}")

    # "accuracy":[],
    # "val_accuracy":[],
    # "loss":[],
    # "val_loss":[]

    metrics_array["loss"].append(metrics['loss'])
    metrics_array["val_loss"].append(metrics['val_loss'])

# import matplotlib.pyplot as plt

# plt.plot(metrics_array["loss"])
# plt.plot(metrics_array["val_loss"])
# plt.show()