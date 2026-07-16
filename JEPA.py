from NeuralNetwork.parameter import Parameter, Simple_Parameter
from NeuralNetwork.neural_network import Layer, LayerNormalization
from Transformer.transformer import Transformer
from Transformer.Embedding import Embedding
from util import mask_tiles, collate_masked_with_unmasked, update_teacher, block_mask_tiles
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cupy as cp
from save_util import save_model

# DATALOADING
X, _ = load_digits(return_X_y=True)
X = cp.expand_dims(cp.asarray(X).reshape(-1, 8, 8), axis=1) / 16
X_train, X_test = train_test_split(X, test_size=0.05)

print(f"Training set full length: {len(X_train)}")
print(f"Testing set full length: {len(X_test)}")

# HYPERPARAMETERS
lr = 0.003
epochs = 300
token_dim = 64 # Token dimensions
stack_dim = 16 # How many patches are there per image?
ratio = 0.6 # What ratio of tokens aren't masked?
masked_dim = int(stack_dim * ratio) # How many patches arent masked?
EMA_ratio = 5e-5 # How much does the student change the teacher?
sample_portion = len(X_train) # How many samples are used?
batch_size = 100

# EMBEDDING
# For now just to get tiling
embedding_space = Embedding(token_dim)

# INIT
# Teacher (Doesnt have a backward because updated by EMA)
ffn1 = Layer(4, 32)
ffn2 = Layer(32, token_dim, 'linear')
p1 = Parameter((stack_dim, token_dim), 0.02)
trans1 = Transformer(token_dim)
teach_ln = LayerNormalization()

# FORWARD
def t_forward(tiles):
    ffn = ffn1.forward(tiles)
    ffn = ffn2.forward(ffn)
    pos = p1.forward(ffn)
    out = trans1.forward(pos)
    return teach_ln.forward(out)

# Student (Updated by unmasked tokens)
ffn3 = Layer(4, 32)
ffn4 = Layer(32, token_dim, 'linear')
p2 = Parameter((stack_dim, token_dim), 0.02)
trans2 = Transformer(token_dim)
stud_ln = LayerNormalization()

def s_forward(tiles, idx):
    ffn = ffn3.forward(tiles)
    ffn = ffn4.forward(ffn)
    pos = p2.forward(ffn, idx)
    out = trans2.forward(pos)
    return stud_ln.forward(out)

def s_backward(d_loss, lr, idx):
    # Only unmasked tokens are used for backprop here
    d_pos = stud_ln.backward(d_loss, lr)
    d_pos = trans2.backward(d_pos, lr)
    d_ffn = p2.backward(d_pos, lr, idx)
    d_ffn = ffn4.backward(d_ffn, lr)
    d_tiles = ffn3.backward(d_ffn, lr)
    return d_tiles

# Predictor (Updated by masked tokens)
masked_p = Simple_Parameter((token_dim,), 0.1)
trans3 = Transformer(token_dim)
p3 = Parameter((stack_dim, token_dim), 0.02)
pred_ln = LayerNormalization()

def p_forward(tokens, visible_idx, masked_idx):
    # Collate masked tokens and known tokens in order here
    mask_token = masked_p.forward()
    full_tokens = collate_masked_with_unmasked((stack_dim, token_dim), tokens, mask_token, visible_idx, masked_idx)
    full_tokens = p3.forward(full_tokens) # Positional encoding for the new token order
    out = trans3.forward(full_tokens)
    return pred_ln.forward(out)

def p_backward(d_loss, lr):
    # Only the masked tokens contribute to this loss
    d_out = trans3.backward(d_loss, lr)
    d_out = p3.backward(d_out, lr)
    return masked_p.backward(d_out, lr)



# =============================
# TRAINING!
# =============================


# Set teacher params to student params
teacher_modules = (ffn1, ffn2, p1, trans1)
student_modules = (ffn3, ffn4, p2, trans2)
update_teacher(teacher_modules, student_modules, 1.0)

# Save random model
save_model(teacher_modules, f"./Checkpoints/teacher_e00.npz")

# Track data
metrics_array = {
    "epoch":[],
    "mse":[],
    "teach":[],
    "mse_val":[],
    "teach_val":[]
}

for epoch in range(epochs):
    running_MSE = 0.0
    running_teacher = 0.0

    running_MSE_val = 0.0
    running_teacher_val = 0.0

    # TRAINING LOOP
    for sample in tqdm(X_train[:sample_portion]):
        # Get tiles for sample
        tiles = embedding_space._tile(sample[0])
        tiles = tiles.reshape(tiles.shape[0], -1) # Flatten regions
        unmasked_tiles, visible_idx, masked_idx = mask_tiles(tiles) # Mask tiles

        # forward pass
        t_out = t_forward(tiles)
        s_out = s_forward(unmasked_tiles, visible_idx)
        p_out = p_forward(s_out, visible_idx, masked_idx)

        # Loss
        diff = p_out[masked_idx] - t_out[masked_idx]
        running_MSE += float(cp.mean(diff ** 2)) 
        running_teacher += float(cp.mean(cp.abs(t_out)))
        d_full = cp.zeros((stack_dim, token_dim), cp.float32) # zero gradient base so I can only pass gradients through masked tokens
        d_full[masked_idx] = 2 * diff / token_dim

        # backward pass
        # Predictor backward
        d_full = pred_ln.backward(d_full, None)
        d_seq = trans3.backward(d_full, lr)
        d_seq = p3.backward(d_seq, lr)
        masked_p.backward(d_seq[masked_idx].sum(axis=0), lr) # ONLY THE SUM OF THE MASKED TOKEN GRADIENTS

        # Student backward
        out = s_backward(d_seq[visible_idx], lr, visible_idx) # ONLY THE NON MASKED TOKEN GRADIENTS

        # Teacher EMA
        update_teacher(teacher_modules, student_modules, EMA_ratio)

    # VALIDATION LOOP
    for sample in X_test:
        # Get tiles for sample
        tiles = embedding_space._tile(sample[0])
        tiles = tiles.reshape(tiles.shape[0], -1) # Flatten regions
        unmasked_tiles, visible_idx, masked_idx = mask_tiles(tiles) # Mask tiles

        # forward pass
        t_out = t_forward(tiles)
        s_out = s_forward(unmasked_tiles, visible_idx)
        p_out = p_forward(s_out, visible_idx, masked_idx)

        # Loss
        diff = p_out[masked_idx] - t_out[masked_idx]
        running_MSE_val += float(cp.mean(diff ** 2)) 
        running_teacher_val += float(cp.mean(cp.abs(t_out)))

    # Verbose and tracker
    print(f"epoch {epoch}  loss {running_MSE / sample_portion:.5f}    teach {running_teacher / sample_portion:.5f}    ",
          f"val_loss {running_MSE_val / len(X_test):.5f}    val_teach {running_teacher_val / len(X_test):.5f}")
    
    metrics_array["epoch"].append(epoch)
    metrics_array["mse"].append(running_MSE / sample_portion)
    metrics_array["teach"].append(running_teacher / sample_portion)
    metrics_array["mse_val"].append(running_MSE_val / len(X_test))
    metrics_array["teach_val"].append(running_teacher_val / len(X_test))

    # Checkpointing
    if (epoch+1) % 10 == 0:
        save_model(teacher_modules, f"./Checkpoints/teacher_e{epoch+1}.npz")

# Save run metrics
import json

with open("output.json", "w") as file:
    json.dump(metrics_array, file)

# Save teacher
save_model(teacher_modules, "teacher.npz")