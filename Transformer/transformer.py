import cupy as cp
from NeuralNetwork.neural_network import LayerNormalization, Layer
from Transformer.multi_head_attn import multi_head_attention

class Transformer:
    def __init__(self, dim, n_heads=4):
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()

        self.mha = multi_head_attention(dim, n_heads)

        self.ffn1 = Layer(dim, dim*4, 'relu')
        self.ffn2 = Layer(dim*4, dim, 'linear')

    def forward(self, x):
        a = self.ln1.forward(x)
        x = x + self.mha.forward(a)
        b = self.ln2.forward(x)
        o = x + self.ffn2.forward(self.ffn1.forward(b))
        return o
    
    def backward(self, d_loss):
        # First loss
        d_ffn2 = self.ffn2.backward(d_loss)
        d_ffn1 = self.ffn1.backward(d_ffn2)
        d_ln2 = self.ln2.backward(d_ffn1)
        d_loss = d_loss + d_ln2

        # First loss
        d_mha = self.mha.backward(d_loss)
        d_ln1 = self.ln1.backward(d_mha)
        d_loss = d_loss + d_ln1
        return d_loss
    
    def step(self, lr, n):
        self.ln1.step(lr, n)
        self.ln2.step(lr, n)
        self.mha.step(lr, n)
        self.ffn1.step(lr, n)
        self.ffn2.step(lr, n)


