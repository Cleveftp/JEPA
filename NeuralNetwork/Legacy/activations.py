import cupy as cp

class ReLU:
    def __init__(self, cuda_device=cp.cuda.Device(0)):
        self.cuda_device = cuda_device

    def forward(self, x):
        with self.cuda_device:
            self.input = x
            return cp.maximum(0,x)
    
    def backward(self, d_loss, _):
        with self.cuda_device:
            return d_loss * (self.input > 0)
    
class Sigmoid:
    def __init__(self, cuda_device=cp.cuda.Device(0)):
        self.cuda_device = cuda_device

    def forward(self, x):
        with self.cuda_device:
            self.input = x
            return 1 / (1 + cp.exp(-x))
    
    def backward(self, d_loss, _):
        with self.cuda_device:
            s = 1 / (1 + cp.exp(-self.input))
            return d_loss * s * (1 - s)
        
class Softmax:
    def __init__(self, cuda_device=cp.cuda.Device(0)):
        self.cuda_device = cuda_device

    def forward(self, x):
        # Stable softmax
        with self.cuda_device:
            shift_x = x - cp.max(x, axis=-1, keepdims=True) # Apparently removing the max here stablizes and prevents massive NaNs 
            exp = cp.exp(shift_x)
            self.A = exp / cp.sum(exp, axis=-1, keepdims=True)
            return self.A
    
    def backward(self, d_loss, _):
        with self.cuda_device:
            dot = cp.sum(d_loss * self.A, axis=-1, keepdims=True)
            return self.A * (d_loss - dot)