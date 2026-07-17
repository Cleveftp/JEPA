import cupy as cp
import cupyx as cpx

class Parameter:
    def __init__(self, dim, init=0.02, cuda_device=cp.cuda.Device(0)):
        # dim describes the shape of the applied param
        # init is a constant to change initial weights
        with cuda_device:
            self.P = cp.random.randn(*dim).astype(cp.float32) * init    # Unusual but necessary

        self.cuda_device = cuda_device

        self.d_loss = cp.zeros_like(self.P)

    def forward(self, x, idx=None):
        if idx is None:
            return x + self.P
        cols = idx[1] if isinstance(idx, tuple) else idx
        return x + self.P[cols]

    def backward(self, d_loss, idx=None):
        if idx is None:
            self.d_loss += d_loss.sum(0) if d_loss.ndim == 3 else d_loss
        else:
            cols = idx[1] if isinstance(idx, tuple) else idx
            cpx.scatter_add(self.d_loss, cols.ravel(), d_loss.reshape(-1, d_loss.shape[-1]))
        return d_loss
    
    def step(self, lr, n):
        self.P -= lr * self.d_loss / n
        self.d_loss = cp.zeros_like(self.P)
        
    
class Simple_Parameter:
    def __init__(self, dim, init=0.02, cuda_device=cp.cuda.Device(0)):
        # dim describes the shape of the applied param
        # init is a constant to change initial weights
        with cuda_device:
            self.P = cp.random.randn(*dim).astype(cp.float32) * init    # Unusual but necessary

        self.cuda_device = cuda_device

        self.d_loss = cp.zeros_like(self.P)

    def forward(self):
        return self.P
    
    def backward(self, d_loss):
        # Parameter doesnt affect the global gradient (return d_loss)
        self.d_loss += d_loss
        return d_loss
    
    def step(self, lr, n):
        self.P -= lr * self.d_loss / n
        self.d_loss = cp.zeros_like(self.P)
