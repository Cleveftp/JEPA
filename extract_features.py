from NeuralNetwork.parameter import Parameter
from NeuralNetwork.neural_network import Layer, LayerNormalization
from Transformer.transformer import Transformer
from save_util import load_model
from Transformer.Embedding import Embedding
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
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
load_model((ffn1, ffn2, p1, trans1), "./Checkpoints/teacher_e300.npz")

def forward(tiles):
    ffn = ffn1.forward(tiles)
    ffn = ffn2.forward(ffn)
    pos = p1.forward(ffn)
    out = trans1.forward(pos)
    return ln1.forward(out)

# Embed and get tokens
embedding_space = Embedding()

# Get dataset
X, _ = load_digits(return_X_y=True)
X = cp.expand_dims(cp.asarray(X).reshape(-1, 8, 8), axis=1) / 16
X_train, X_test = train_test_split(X, test_size=0.05)

reps = []
for sample in X_test:
    tiles = embedding_space._tile(sample[0])
    tiles = tiles.reshape(tiles.shape[0], -1)
    reps.append(forward(tiles))

test_reps = cp.array(reps)
print(test_reps.shape)

reps = []
for sample in X_train:
    tiles = embedding_space._tile(sample[0])
    tiles = tiles.reshape(tiles.shape[0], -1)
    reps.append(forward(tiles))

train_reps = cp.array(reps)
print(train_reps.shape)

cp.save("test_reps.npy",test_reps)
cp.save("train_reps.npy",train_reps)