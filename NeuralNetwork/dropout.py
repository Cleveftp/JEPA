import cupy as cp

class Dropout:
    def __init__(self, d=0.10, cuda_device=cp.cuda.Device(0)):
        self.cuda_device = cuda_device
        self.d = d

    def forward(self, x):
        with self.cuda_device:
            mask = cp.random.rand(*x.shape) < self.d # Create a mask of % d True (I didnt know this but apparently * unpacks tuples into their values)
            x[mask] = 0
            return x
    
    def backward(self, d_loss):
        return d_loss
    
    def step(self, lr=None, n=None):
        pass