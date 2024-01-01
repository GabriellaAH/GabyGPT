import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

# Hyperparameters for the model
batch_size = 16
block_size = 128
max_iters = 5000
learning_rate = 4e-4
eval_iters = 500
n_embd = 1140
n_head = 32
n_layer = 36
dropout = 0.2
load_model = False
load_model_name = 'gabyGPT-00.pkl'
save_model_name = 'gabyGPT-00.pkl'

print(device)

# Load and process the vocabulary
tokenizer = ByteLevelBPETokenizer("./BPEVocab/vocab.json", "./BPEVocab/merges.txt")
vocab_size = 45000

encode = lambda text: tokenizer.encode(text).ids
decode = lambda token_ids: tokenizer.decode(token_ids)

# Function to get a random chunk of text from the dataset
def get_random_chunk(split):
    filename = "./output_train.txt" if split == 'train' else "./output_val.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data

# Function to create a batch for training/validation
def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Function to estimate the loss for training and validation sets
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Definition of a single head in the multi-head attention mechanism
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # Key
        q = self.query(x)  # Query
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # Scaled dot-product attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Masking for causal (unidirectional) attention
        wei = F.softmax(wei, dim=-1)  # Softmax over the last dimension
        wei = self.dropout(wei)
        v = self.value(x)  # Value
        out = wei @ v  # Output of the attention mechanism
        return out

# Multi-head attention combines several heads
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate the output of all heads
        out = self.dropout(self.proj(out))  # Project back to the embedding dimension
        return out

# Feedforward network used in each transformer block
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# The Block class represents one block of the Transformer model
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        y = self.ln1(x)
        y = self.sa(y)
        x = x + y  # Apply residual connection
        y = self.ln2(x)
        y = self.ffwd(y)
        x = x + y  # Apply another residual connection
        return x

# GPTLanguageModel defines the overall language model architecture
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Initialize weights for linear and embedding layers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)  # Token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Positional embeddings
        x = tok_emb + pos_emb  # Sum token and position embeddings

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)  # Apply final layer normalization
        logits = self.lm_head(x)  # Project back to vocabulary size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # Calculate cross-entropy loss

        return logits, loss

    def generate(self, index, max_new_tokens):
        # Generate text given an input index
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :]  # Take the last step
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            index = torch.cat((index, index_next), dim=1)  # Append to the sequence
        return index

# Initialize the model and optionally load pre-trained weights
model = GPTLanguageModel(vocab_size)
if load_model:
    print('loading model parameters...')
    # with open(load_model_name, 'rb') as f:
    #     model = pickle.load(f)    
    model.load_state_dict(torch.load(load_model_name))
    print('loaded successfully!')

# Count the number of trainable parameters
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_parameters)

# Move the model to the appropriate device (GPU/CPU)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for iter in tqdm(range(max_iters), total=max_iters):
    if iter % eval_iters == 0 and iter > 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    xb, yb = get_batch('train')

    # Forward pass and compute loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# Save the trained model
torch.save(model.state_dict(), save_model_name)
print('model saved')
