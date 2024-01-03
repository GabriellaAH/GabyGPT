import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

# Hyperparameters for the model
batch_size = 12
block_size = 128
max_iters = 25000
learning_rate = 3e-4
eval_iters = 1000
n_embd = 1140
n_head = 32
n_layer = 36
dropout = 0.2
load_model = True
load_model_name = 'gabyGPT-05.pkl'
save_model_name = 'gabyGPT-06.pkl'

print(device)

# Load and process the vocabulary
tokenizer = ByteLevelBPETokenizer("./BPEVocab/vocab.json", "./BPEVocab/merges.txt")
vocab_size = 45000

encode = lambda text: tokenizer.encode(text).ids
decode = lambda token_ids: tokenizer.decode(token_ids)

class TextDataset(Dataset):
    def __init__(self, file_path, block_size, tokenizer, cache_size=batch_size*2):
        self.file_path = file_path
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.cache = []
        self.cache_size = cache_size
        self._fill_cache()        
        self.num_samples = self._get_num_samples()

    def _fill_cache(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for _ in range(self.cache_size):
                line = next(file, None)
                if line is None: break
                self.cache.append(self.tokenizer(line.strip()))
                
    def _get_num_samples(self):
        # Estimate the number of samples in the file
        with open(self.file_path, 'r', encoding='utf-8') as file:
            count = sum(1 for _ in file)
        return count // self.block_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= len(self.cache):
            self._fill_cache()
        return torch.tensor(self.cache[idx % self.cache_size], dtype=torch.long)

        # with open(self.file_path, 'r', encoding='utf-8') as file:
        #     for i, line in enumerate(file):
        #         if i == idx:
        #             encoded_chunk = self.tokenizer(line.strip())
        #             break
        # encoded_tensor = torch.tensor(encoded_chunk, dtype=torch.long)
        # return encoded_tensor

    
def get_random_batch(data_loader, num_samples=100, pad_length=128):
    random_indices = torch.randint(0, len(data_loader.dataset), (num_samples,))
    random_batches = [data_loader.dataset[i] for i in random_indices]

    padded_batches = [F.pad(batch, (0, pad_length - len(batch)), mode='constant', value=0) for batch in random_batches]
    batch = torch.stack(padded_batches).to(device)

    # Prepare input and target tensors
    batch_inputs = batch[:, :-1] 
    batch_targets = batch[:, 1:] 

    return batch_inputs, batch_targets
@torch.no_grad()
def estimate_loss(train_data_loader, val_data_loader, num_samples=100):
    """
    Estimates the loss for both training and validation sets.

    Returns:
        dict: A dictionary with mean loss values for 'train' and 'val' splits.
    """    
    out = {}
    model.eval()
    for split, data_loader in [('train', train_data_loader), ('val', val_data_loader)]:
        losses = []
        for _ in range(num_samples):
            X, Y = get_random_batch(data_loader)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses.append(loss.item())
            del X, Y
        out[split] = torch.tensor(losses).mean().item()
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
        """
        Forward pass for a single attention head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the attention mechanism.
        """        
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
        """
        Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying multi-head attention.
        """        
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
        """
        Forward pass for the feedforward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the feedforward network.
        """        
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
        """
        Forward pass for a single Transformer block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the Transformer block.
        """        
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
        """
        Initializes weights for the linear and embedding layers of the model.

        This method applies a specific weight initialization strategy to different types of layers.
        For linear layers, it initializes the weights from a normal distribution with a mean of 0.0 and a standard deviation of 0.02.
        If the linear layer has a bias term, it initializes the bias to zero.
        For embedding layers, it initializes the weights from a normal distribution with a mean of 0.0 and a standard deviation of 0.02.

        Args:
            module (nn.Module): A PyTorch module from the model, typically a layer.

        Note:
            This method is designed to be used with the `apply` method of `nn.Module`, which applies a function recursively to every submodule.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        """
        Forward pass for the GPT language model.

        Args:
            index (torch.Tensor): Tensor of token indices.
            targets (torch.Tensor, optional): Target token indices for loss calculation.

        Returns:
            tuple: A tuple containing logits and loss (if targets are provided).
        """        
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
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)  # Calculate cross-entropy loss

        return logits, loss

    def generate(self, index, max_new_tokens):
        """
        Generates text given an input index.

        Args:
            index (torch.Tensor): Tensor of starting token indices.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            torch.Tensor: Tensor of token indices including the generated tokens.
        """
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

train_ds = TextDataset("./output_train.txt", block_size, tokenizer=encode)
train_data_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)

val_ds = TextDataset("./output_val.txt", block_size, tokenizer=encode)
val_data_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=10)

# Training loop
model.train()
for iter in tqdm(range(max_iters), total=max_iters):
    if iter % eval_iters == 0 and iter > 0:
        del xb, yb, logits, loss
        torch.cuda.empty_cache()        
        losses = estimate_loss(train_data_loader, val_data_loader)
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
        torch.save(model.state_dict(), f'{load_model_name}_and_{iter}.pkl')

    xb, yb = get_random_batch(train_data_loader, batch_size, pad_length=block_size)

    # Forward pass and compute loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# Save the trained model
torch.save(model.state_dict(), save_model_name)
print('model saved')
