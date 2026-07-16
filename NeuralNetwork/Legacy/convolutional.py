import cupy as cp
from cupy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.datasets import load_digits

class Conv_2D:
    def __init__(self, kernel_size, out_channels, in_channels, cuda_device=cp.cuda.Device(0)):
        with cuda_device:
            # Initialize kernel with channels
            fan_in = in_channels * kernel_size * kernel_size
            self.kernel = cp.random.randn(out_channels, in_channels, kernel_size, kernel_size) * cp.sqrt(2.0 / fan_in) # HE INITIALIZATION
            self.bias = cp.zeros((out_channels,), dtype=cp.float32)

        self.cuda_device = cuda_device
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        with self.cuda_device:
            # channels, height, width dont pad channels
            x_pad = cp.pad(x, pad_width=((0,0), (1,1), (1,1)))

            # Get output size
            _, self.x_height, self.x_width = x_pad.shape
            k = int(self.kernel_size)

            out_h = self.x_height - k + 1
            out_w = self.x_width - k + 1

            # transformation along x
            sc, sh, sw = x_pad.strides

            # Dont understand this but it strides well 
            # (maybe makes windows based on image and the tensordot just applies 
            # the kernel to every window to create the output and reshapes it?)
            self.windows = as_strided(x_pad, 
                                shape=(out_h, out_w, self.in_channels, k, k), 
                                strides=(sh, sw, sc, sh, sw))
            
            out = cp.tensordot(self.windows, self.kernel, axes=((2, 3, 4), (1, 2, 3)))
            out += self.bias # Overall gradient bias

        return cp.transpose(out, (2, 0, 1))
    
    def backward(self, d_loss, lr):
        # Start with the gradient update based on weights
        with self.cuda_device:
            # The kernel size
            k = int(self.kernel_size)

            # Runs the loss over the output and resizes it
            dW = cp.tensordot(d_loss, self.windows, axes=((1, 2), (0, 1)))
            dB = cp.sum(d_loss, axis=(1, 2))

            # Pad the loss map
            d_loss_pad = cp.pad(d_loss, pad_width=((0, 0), (k - 1, k - 1), (k - 1, k - 1)))
            dc, dsh, dsw = d_loss_pad.strides # Checks how many strides there are up and across on the padded loss

            # Create windows for the loss with the gradient map
            d_loss_windows = as_strided(d_loss_pad, 
                                        shape=(self.x_height, self.x_width, self.out_channels, k, k), 
                                        strides=(dsh, dsw, dc, dsh, dsw))
            flipped_kernel = cp.rot90(self.kernel, 2, axes=(2, 3)) # Rotate kernel

            # Apply the flipped kernel to the windows
            d_x_pad = cp.tensordot(d_loss_windows, flipped_kernel, axes=((2, 3, 4), (0, 2, 3)))

            # Remove padding on the output gradient 
            d_input = cp.transpose(d_x_pad, (2, 0, 1))[:, 1:-1, 1:-1]

            # Update kernel w and b
            self.kernel -= lr * dW
            self.bias -= lr * dB

        return d_input

class Flatten:
    def __init__(self):
        pass

    def forward(self, x):
        self.shape = x.shape
        return x.ravel()
    
    def backward(self, d_loss, _):
        return d_loss.reshape(self.shape)
    
class Reshape:
    def __init__(self):
        pass

    def forward(self, x, shape):
        self.shape = x.shape
        return x.reshape(shape)
    
    def backward(self, d_loss, _):
        return d_loss.reshape(self.shape)
        
if __name__ == "__main__":
    from NeuralNetwork.Legacy.neural_network import Sequential, Layer
    from NeuralNetwork.Legacy.activations import ReLU, Sigmoid

    model = Sequential(0.1)

    model.add_layers([
        Conv_2D(3, 1, 1),
        ReLU(),
        Flatten(),
        Layer(100, 10, 'sigmoid')
    ])

    target = cp.random.rand(10)
    inputs = cp.random.rand(1, 10, 10)

    for i in range(200):
        pred = model.forward(inputs)

        d_mse = (2 * (pred - target)) / inputs.size
        loss = cp.mean(cp.square(pred - target).get())
        print(f"Epoch {i}: {loss:4f}")

        model.backward(d_mse)

    print(pred.shape)