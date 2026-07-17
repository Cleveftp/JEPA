from NeuralNetwork.parameter import Parameter, Simple_Parameter
from NeuralNetwork.neural_network import Layer, LayerNormalization
from Transformer.transformer import Transformer
from util import mask_tiles, collate_masked_with_unmasked, update_teacher, get_batches, tile_batch
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
lr = 0.01
epochs = 500
token_dim = 64 # Token dimensions
stack_dim = 16 # How many patches are there per image?
ratio = 0.6 # What ratio of tokens aren't masked?
masked_dim = int(stack_dim * ratio) # How many patches arent masked?
EMA_ratio = 1e-5 # How much does the student change the teacher?
sample_portion = len(X_train) # How many samples are used?
batch_size = 16

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

def s_backward(d_loss, idx):
    # Only unmasked tokens are used for backprop here
    d_pos = stud_ln.backward(d_loss)
    d_pos = trans2.backward(d_pos)
    d_ffn = p2.backward(d_pos, idx)
    d_ffn = ffn4.backward(d_ffn)
    d_tiles = ffn3.backward(d_ffn)
    return d_tiles

def s_step(lr, batch_len):
    stud_ln.step(lr, batch_len)
    trans2.step(lr, batch_len)
    p2.step(lr, batch_len)
    ffn4.step(lr, batch_len)
    ffn3.step(lr, batch_len)

# Predictor (Updated by masked tokens)
masked_p = Simple_Parameter((token_dim,), 0.1)
trans3 = Transformer(token_dim)
p3 = Parameter((stack_dim, token_dim), 0.02)
pred_ln = LayerNormalization()

def p_forward(tokens, visible_idx, masked_idx):
    # Collate masked tokens and known tokens in order here
    mask_token = masked_p.forward()
    full_tokens = collate_masked_with_unmasked((tokens.shape[0], stack_dim, token_dim), tokens, mask_token, visible_idx, masked_idx)
    full_tokens = p3.forward(full_tokens) # Positional encoding for the new token order
    out = trans3.forward(full_tokens)
    return pred_ln.forward(out)

# BACKWARD COMPUTED IN LOOP

def p_step(lr, batch_len):
    masked_p.step(lr, batch_len)
    trans3.step(lr, batch_len)
    p3.step(lr, batch_len)
    pred_ln.step(lr, batch_len) # Technically does nothing but might be useful later

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

    # Collate batches first
    batches = get_batches(X_train[:sample_portion], batch_size)

    for batch in tqdm(batches):
        B = batch.shape[0]
        # Get tiles for sample
        tiles = tile_batch(batch)
        batch_tiles, visible_idx, masked_idx = mask_tiles(tiles, ratio) # Create masks

        # forward pass
        t_out = t_forward(tiles)
        s_out = s_forward(batch_tiles, visible_idx)
        p_out = p_forward(s_out, visible_idx, masked_idx)

        # Loss
        diff = p_out[masked_idx] - t_out[masked_idx]
        running_MSE += float(cp.mean(diff ** 2)) 
        running_teacher += float(cp.mean(cp.abs(t_out)))
        d_full = cp.zeros((B, stack_dim, token_dim), cp.float32) # zero gradient base so I can only pass gradients through masked tokens
        d_full[masked_idx] = 2 * diff / token_dim

        # backward pass
        # Predictor backward
        d_full = pred_ln.backward(d_full)
        d_seq = trans3.backward(d_full)
        d_seq = p3.backward(d_seq)
        masked_p.backward(d_seq[masked_idx].sum(axis=(0,1))) # ONLY THE SUM OF THE MASKED TOKEN GRADIENTS

        # Student backward
        out = s_backward(d_seq[visible_idx], visible_idx) # ONLY THE NON MASKED TOKEN GRADIENTS

        # Step all components
        s_step(lr, B)
        p_step(lr, B)

        # Teacher EMA
        update_teacher(teacher_modules, student_modules, EMA_ratio)

    # VALIDATION LOOP
    tiles = tile_batch(X_test)
    batch_tiles, visible_idx, masked_idx = mask_tiles(tiles, ratio)

    t_out = t_forward(tiles)
    s_out = s_forward(batch_tiles, visible_idx)
    p_out = p_forward(s_out, visible_idx, masked_idx)

    diff = p_out[masked_idx] - t_out[masked_idx]
    running_MSE_val   = float(cp.mean(diff ** 2))
    running_teacher_val = float(cp.mean(cp.abs(t_out)))

    # Verbose and tracker
    print(f"epoch {epoch}  loss {running_MSE / len(batches):.5f}    teach {running_teacher / len(batches):.5f}    ",
          f"val_loss {running_MSE_val:.5f}    val_teach {running_teacher_val:.5f}")
    
    metrics_array["epoch"].append(epoch)
    metrics_array["mse"].append(running_MSE / len(batches))
    metrics_array["teach"].append(running_teacher / len(batches))
    metrics_array["mse_val"].append(running_MSE_val)
    metrics_array["teach_val"].append(running_teacher_val)

    # Checkpointing
    if (epoch+1) % 10 == 0:
        save_model(teacher_modules, f"./Checkpoints/teacher_e{epoch+1}.npz")

# Save run metrics
import json

with open("output.json", "w") as file:
    json.dump(metrics_array, file)

# Save teacher
save_model(teacher_modules, "teacher.npz")