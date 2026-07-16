import cupy as cp

class Layer:
    def __init__(self, input_size, output_size, activation='relu', cuda_device=cp.cuda.Device(0)):
        """Needs weights, bias, store input, store forward pass output, activation layer"""
        with cuda_device:
            self.W = cp.random.randn(output_size, input_size).astype(cp.float32) * cp.sqrt(2.0 / input_size)
            self.B = cp.zeros(output_size, dtype=cp.float32)

        # CUDA SUPPORT
        self.activation = activation.lower()
        self.cuda_device = cuda_device

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

    def backward(self, d_loss, lr):
        # dL/dw = dL/dz * dz/dx * dx/dw
        with self.cuda_device:
            dL_dz = d_loss * self._d_activate(self.out)

            if self.input.ndim == 1:
                dW = cp.outer(dL_dz, self.input)
                dB = dL_dz
            else:
                dW = dL_dz.T @ self.input
                dB = dL_dz.sum(axis=0)
            
            # Gradients for next layer
            d_input = cp.dot(dL_dz, self.W)

            # Weight update w_n = w_o - lr*dL/dw
            # Weight and Bias update
            self.W -= lr * dW
            self.B -= lr * dB

            return d_input
        
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
            grad = layer.backward(grad, self.lr)

class LayerNormalization:
    def __init__(self, epsilon=1e-5):
        self.eps = epsilon

    def forward(self, x):
        self.mean = cp.mean(x, axis=-1, keepdims=True)
        self.var = cp.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / cp.sqrt(self.var + self.eps)
        return self.x_hat

    def backward(self, d_loss, _):
        # Normalization gradient tracking
        N = d_loss.shape[-1]
        dx = (N * d_loss - cp.sum(d_loss, axis=-1, keepdims=True) - self.x_hat * cp.sum(d_loss * self.x_hat, axis=-1, keepdims=True)) / (N * cp.sqrt(self.var + self.eps))
        return dx


if __name__ == "__main__":
    model = Sequential(lr=0.01)
    model.add_layers([
        Layer(10, 64, activation='sigmoid'),
        Layer(64, 128, activation='sigmoid'),
        Layer(128, 5, activation='sigmoid')
    ])

    target = cp.random.rand(5)
    inputs = cp.random.randn(10)

    def mse(y, y_):
        loss = cp.mean(cp.square(y_ - y), dtype=cp.float16)
        grad = -2 * (y_ - y) / y.size 
        return loss, grad
    
    for i in range(10000):
        pred = model.forward(inputs)
        loss, d_loss = mse(pred, target)

        if i % 100 == 0:
            print(f"Epoch {i}: {cp.average(loss)}")
        model.backward(d_loss)
    
