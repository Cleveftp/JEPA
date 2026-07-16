import cupy as cp
from NeuralNetwork.activations import Softmax

class self_attn:
    def __init__(self, input_dim=128, head_dim=128, cuda_device=cp.cuda.Device(0)):
        with cuda_device:
            scale = cp.sqrt(2.0 / input_dim)
            self.wQ = (cp.random.randn(input_dim, head_dim) * scale).astype(cp.float32)
            self.wK = (cp.random.randn(input_dim, head_dim) * scale).astype(cp.float32)
            self.wV = (cp.random.randn(input_dim, head_dim) * scale).astype(cp.float32)

        self.cuda_device = cuda_device
        self.head_dim = head_dim

        self.softmax = Softmax()

        self.dWQ = cp.zeros_like(self.wQ)
        self.dWK = cp.zeros_like(self.wK)
        self.dWV = cp.zeros_like(self.wV)

    def forward(self, x):
        self.x = x
        self.Q = cp.dot(x, self.wQ)
        self.K = cp.dot(x, self.wK)
        self.V = cp.dot(x, self.wV)

        self.S = cp.dot(self.Q, self.K.T) / cp.sqrt(self.head_dim)

        self.A = self.softmax.forward(self.S)

        self.O = cp.dot(self.A, self.V)
        return self.O
    
    def backward(self, d_loss):
        dV = cp.dot(self.A.T, d_loss)
        dA = cp.dot(d_loss, self.V.T)

        dS = self.softmax.backward(dA, None) / cp.sqrt(self.head_dim)
        dQ = cp.dot(dS, self.K)
        dK = cp.dot(dS.T, self.Q)

        dWQ = cp.dot(self.x.T, dQ)
        dWK = cp.dot(self.x.T, dK)
        dWV = cp.dot(self.x.T, dV)

        dX = cp.dot(dQ, self.wQ.T) + cp.dot(dK, self.wK.T) + cp.dot(dV, self.wV.T)

        self.dWQ += dWQ
        self.dWK += dWK
        self.dWV += dWV

        return dX
    
    def step(self, lr, n):
        self.wQ -= lr * self.dWQ / n
        self.wK -= lr * self.dWK / n
        self.wV -= lr * self.dWV / n
        self.dWQ = cp.zeros_like(self.wQ)
        self.dWK = cp.zeros_like(self.wK)
        self.dWV = cp.zeros_like(self.wV)

if __name__ == "__main__":
    input_tokens = cp.random.randn(5, 16)

    attn = self_attn(input_dim=16, head_dim=16)
    out = attn.forward(input_tokens)
    print(out)