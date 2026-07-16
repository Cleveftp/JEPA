from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from NeuralNetwork.neural_network import Sequential, Layer
from NeuralNetwork.dropout import Dropout
import numpy as np
import cupy as cp
from tqdm import tqdm


feats = np.load("Downstream/Features/features_second.npy")
y_raw = np.load("Downstream/Features/labels_second.npy")       
                            # (1797,)
Y = OneHotEncoder(sparse_output=False).fit_transform(y_raw.reshape(-1, 1))
X = feats.reshape(len(feats), -1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=y_raw) 

X_train, X_test = cp.asarray(X_train, cp.float32), cp.asarray(X_test, cp.float32)
y_train, y_test = cp.asarray(y_train, cp.float32), cp.asarray(y_test, cp.float32)

print(X_train.shape)

# HYPERPARAMS
lr = 0.01
epochs = 50

model = Sequential(lr)

model.add_layers([
    Layer(len(X_train[0]), 256),
    Layer(256, 512),
    Dropout(0.05),
    Layer(512, 32),
    Layer(32, 10, 'sigmoid'),
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