from NeuralNetwork.parameter import Parameter
from NeuralNetwork.neural_network import Layer, LayerNormalization
from Transformer.transformer import Transformer
from save_util import load_model
from util import tile_batch
from sklearn.datasets import load_digits
import cupy as cp

# Model details
token_dim = 64
stack_dim = 16

# Load model
ffn1 = Layer(4, 32) 
ffn2 = Layer(32, token_dim, 'linear')
p1 = Parameter((stack_dim, token_dim)) 
trans1 = Transformer(token_dim)
ln1 = LayerNormalization()
load_model((ffn1, ffn2, p1, trans1), "./Checkpoints/teacher_e500.npz")

def forward(tiles):
    ffn = ffn1.forward(tiles)
    ffn = ffn2.forward(ffn)
    pos = p1.forward(ffn)
    out = trans1.forward(pos)
    return ln1.forward(out)

# Get dataset
X, y = load_digits(return_X_y=True)
imgs  = cp.asarray(X).reshape(-1, 1, 8, 8) / 16.0
tiles = tile_batch(imgs)
reps  = forward(tiles)

cp.save("features.npy", reps.astype(cp.float32))
cp.save("labels.npy", y)