import cupy as cp
from Transformer.self_attention import self_attn

class multi_head_attention:
    def __init__(self, dim, n_heads=4):
        self.head_dim = dim // n_heads 
        self.heads = [self_attn(input_dim=dim, head_dim=self.head_dim) for _ in range(n_heads)]
        self.wO = (cp.random.randn(n_heads*self.head_dim, dim) * cp.sqrt(2/dim)).astype(cp.float32)

    def forward(self, x):
        # Run all inputs through all heads
        outs = [head.forward(x) for head in self.heads]
        self.concat = cp.concatenate(outs, axis=1)
        self.x = x
        return cp.dot(self.concat, self.wO)
    
    def backward(self, d_loss, lr):
        d_concat = cp.dot(d_loss,self.wO.T)
        dWO = cp.dot(self.concat.T, d_loss)
        dX = 0

        for i, head in enumerate(self.heads):
            chunk = d_concat[:, i*self.head_dim:(i+1)*self.head_dim] # Split
            dX += head.backward(chunk, lr)

        self.wO -= lr * dWO
        return dX
