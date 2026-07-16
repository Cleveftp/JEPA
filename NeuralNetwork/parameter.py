import cupy as cp

class Parameter:
    def __init__(self, dim, init=0.02, cuda_device=cp.cuda.Device(0)):
        # dim describes the shape of the applied param
        # init is a constant to change initial weights
        with cuda_device:
            self.P = cp.random.randn(*dim).astype(cp.float32) * init    # Unusual but necessary

        self.cuda_device = cuda_device

    def forward(self, x, idx=None):
        # Allows me to have a fixed parameter table without dropping masked parameters
        return x + (self.P if idx is None else self.P[idx])

    def backward(self, d_loss, lr, idx=None):
        if idx is None:
            self.P -= lr * d_loss
        else:
            self.P[idx] -= lr * d_loss
        return d_loss
    
class Simple_Parameter:
    def __init__(self, dim, init=0.02, cuda_device=cp.cuda.Device(0)):
        # dim describes the shape of the applied param
        # init is a constant to change initial weights
        with cuda_device:
            self.P = cp.random.randn(*dim).astype(cp.float32) * init    # Unusual but necessary

        self.cuda_device = cuda_device

    def forward(self):
        # Simple aggregation
        return self.P  
    
    def backward(self, d_loss, lr):
        # Parameter doesnt affect the global gradient (return d_loss)
        self.P -=  d_loss * lr
        return d_loss
