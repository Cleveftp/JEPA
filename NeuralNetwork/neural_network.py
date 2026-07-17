import cupy as cp

class Layer: # Can handle batches
    def __init__(self, input_size, output_size, activation='relu', cuda_device=cp.cuda.Device(0)):
        """Needs weights, bias, store input, store forward pass output, activation layer"""
        with cuda_device:
            self.W = cp.random.randn(output_size, input_size).astype(cp.float32) * cp.sqrt(2.0 / input_size)
            self.B = cp.zeros(output_size, dtype=cp.float32)

        # CUDA SUPPORT
        self.activation = activation.lower()
        self.cuda_device = cuda_device

        # For cumulative gradient
        self.dW = cp.zeros_like(self.W)
        self.dB = cp.zeros_like(self.B)

    def _activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + cp.exp(-x))
        elif self.activation == 'relu':
            return cp.maximum(0,x)
        elif self.activation == 'linear':
            return x
        else:
            return cp.maximum(0, x)

    def _d_activate(self, x):
        if self.activation == 'sigmoid':
            s = 1 / (1 + cp.exp(-x))
            return s * (1 - s)
        elif self.activation == 'relu':
            return (x > 0).astype(x.dtype)
        elif self.activation == 'linear':
            return cp.ones_like(x)
        else:
            return (x > 0).astype(x.dtype)
        
    def forward(self, x):
        # Forward pass y = wx + b
        with self.cuda_device:
            # Store input
            self.input = x
            self.out = cp.dot(x, self.W.T) + self.B
        return self._activate(self.out)

    def backward(self, d_loss):
        # dL/dw = dL/dz * dz/dx * dx/dw
        with self.cuda_device:
            dL_dz = d_loss * self._d_activate(self.out)
            x2 = self.input.reshape(-1, self.input.shape[-1]) # Reshape along batch
            g2 = dL_dz.reshape(-1, dL_dz.shape[-1]) # Reshape output
            self.dW += cp.dot(g2.T, x2) 
            self.dB += g2.sum(0) 
            return cp.dot(dL_dz, self.W)

    def step(self, lr, n): # Averaged step based on broader context (batch size)
        self.W -= lr * self.dW / n
        self.B -= lr * self.dB / n
        self.dW = cp.zeros_like(self.W)
        self.dB = cp.zeros_like(self.B)
        
class Sequential:
    def __init__(self, lr):
        self.lr = lr
        self.layers = []

    def add_layers(self, layers):
        # More layers
        self.layers = layers
        
    def forward(self, x):
        # Get output from layers
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, d_loss):
        # Backpropagate from loss
        grad = d_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, n):
        for layer in self.layers:
            layer.step(self.lr, n)

class LayerNormalization:
    def __init__(self, epsilon=1e-5):
        self.eps = epsilon

    def forward(self, x):
        self.mean = cp.mean(x, axis=-1, keepdims=True)
        self.var = cp.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / cp.sqrt(self.var + self.eps)
        return self.x_hat

    def backward(self, d_loss):
        # Normalization gradient tracking
        N = d_loss.shape[-1]
        dx = (N * d_loss - cp.sum(d_loss, axis=-1, keepdims=True) - self.x_hat * cp.sum(d_loss * self.x_hat, axis=-1, keepdims=True)) / (N * cp.sqrt(self.var + self.eps))
        return dx
    
    def step(self, lr=None, n=None):
        pass


if __name__ == "__main__":
    cp.random.seed(42)

    BATCH_SIZE = 10
    LR = 0.1
    EPOCHS = 200

    model = Sequential(lr=LR)
    model.add_layers([
        Layer(10, 64, activation='sigmoid'),
        Layer(64, 128, activation='sigmoid'),
        Layer(128, 5, activation='sigmoid')
    ])

    target = cp.random.rand(200,5)
    inputs = cp.random.randn(200,10)

    def mse(y, y_):
        loss = cp.mean(cp.square(y_ - y))
        grad = -2 * (y_ - y) / y.shape[1]
        return loss, grad
    
    def get_batches(arr, targ, batch_size, axis=0):
        n_chunks = -(-arr.shape[axis] // batch_size)
        return cp.array_split(arr, n_chunks, axis=axis), cp.array_split(targ, n_chunks, axis=axis)
    
    for e in range(EPOCHS):
        cum_loss = 0.0
        batches, batch_targets = get_batches(inputs, target, BATCH_SIZE)
        for batch, targ in zip(batches, batch_targets):
            pred = model.forward(batch)
            loss, d_loss = mse(pred, targ)
            cum_loss += loss
            model.backward(d_loss)
            model.step(BATCH_SIZE)

        print(f"Epoch {e}: {cp.average(cum_loss)}")
        
        
    
