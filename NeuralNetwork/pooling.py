import cupy as cp

class MaxPool:
    def __init__(self, pool_size, cuda_device=cp.cuda.Device(0)):
        self.p = pool_size
        self.cuda_device = cuda_device

    def forward(self, x):
        with self.cuda_device:
            # Get shape
            self.input_shape = x.shape
            c, h, w = x.shape

            # Reshape x to the pooled shape
            self.x = x.reshape(c, h // self.p, self.p, w // self.p, self.p)
            self.out = cp.max(self.x, axis=(2,4)) # replace regions with max

            return self.out
    
    def backward(self, d_loss):
        with self.cuda_device:
            # Rebuild original dims to find where the kernel created maximums
            out_expanded = cp.expand_dims(cp.expand_dims(self.out, 2), 4)
            mask = (self.x == out_expanded).astype(d_loss.dtype)

            # Expand dims across gradient to be the same shape as the input
            d_loss_expanded = cp.expand_dims(cp.expand_dims(d_loss, 2), 4)
            d_input_reshaped = mask * d_loss_expanded # Apply gradient to pixel
            
            return d_input_reshaped.reshape(self.input_shape) # Return to original dims
        
    def step(self, lr=None, n=None):
        pass
        
class Upsample:
    def __init__(self, scale=2, cuda_device=cp.cuda.Device(0)):
        self.s = scale
        self.cuda_device = cuda_device

    def forward(self, x):
        with self.cuda_device:
            # Get shape
            self.input_shape = x.shape

            # Create upsampled tiles
            out = cp.repeat(cp.repeat(x, self.s, axis=1), self.s, axis=2)
            return out
    
    def backward(self, d_loss):
        with self.cuda_device:
            c, h, w = self.input_shape

            # Return to original size
            d_loss_reshaped = d_loss.reshape(c, h, self.s, w, self.s)
            d_input = cp.sum(d_loss_reshaped, axis=(2,4)) # Combine the gradients in the blocks

            return d_input
        
    def step(self, lr=None, n=None):
        pass
        
class Concatenate:
    def __init__(self, axis=0, cuda_device=cp.cuda.Device(0)):
        self.axis = axis
        self.cuda_device = cuda_device

    def forward(self, x, y):
        with self.cuda_device:
            self.x_shape = x.shape
            return cp.concatenate((x,y), axis=self.axis)
    
    def backward(self, d_loss):
        with self.cuda_device:
            split_indices = [self.x_shape[self.axis]]
            d_x, d_y = cp.split(d_loss, split_indices, axis=self.axis)
            return d_x, d_y
        
    def step(self, lr=None, n=None):
        pass
        
class Reshape:
    def __init__(self, target_shape, cuda_device=cp.cuda.Device(0)):
        self.target_shape = target_shape
        self.cuda_device = cuda_device

    def forward(self, x):
        with self.cuda_device:
            self.input_shape = x.shape
            return cp.reshape(x, self.target_shape)
    
    def backward(self, d_loss):
        with self.cuda_device:
            return cp.reshape(d_loss, self.input_shape)
        
    def step(self, lr=None, n=None):
        pass