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
from collections import OrderedDict
import torch
from torch.utils.data import Sampler
import random
from pathlib import Path
import os
import struct

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

# Hyperparameters for the model
batch_size = 10
block_size = 128
max_iters = 5010
learning_rate = 3e-4
eval_iters = 5000
n_embd = 1140
n_head = 32
n_layer = 36
dropout = 0.2
load_model = True
window_size = 128
load_model_name = 'gabyGPT-22.pkl'
save_model_name = 'gabyGPT-23.pkl'

train_file = './output_train.txt'
# train_file = './w.txt'
val_file = './output_val.txt'

print(device)

# Load and process the vocabulary
tokenizer = ByteLevelBPETokenizer("./BPEVocab/vocab.json", "./BPEVocab/merges.txt")
vocab_size = 45000

encode = lambda text: tokenizer.encode(text).ids
decode = lambda token_ids: tokenizer.decode(token_ids)

def create_token_file(input_file_path, output_file_path):
    total_lines = sum(1 for _ in open(input_file_path, 'r', encoding='utf-8'))
    
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'wb') as output_file:
        for line in tqdm(input_file, total=total_lines, desc="Processing file"): 
            token_ids = tokenizer.encode(line).ids
            for token_id in token_ids:
                output_file.write(struct.pack('I', token_id))                
        
def create_index_table_if_not_exist(file_path):
    file_idx = Path(f"{file_path}.pkl")
    if not file_idx.is_file():
        create_token_file(file_path, f"{file_path}.pkl")

class SequentialBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            yield range(i, min(i + self.batch_size, len(self.dataset)))

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    

class TextDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.file_path = file_path
        self.block_size = block_size
        self.num_samples = self._get_num_samples()
        self.file = open(self.file_path, 'rb')

    def _get_num_samples(self):
        with open(self.file_path, 'rb') as file:
            file.seek(0, 2)  # Move to the end of the file
            file_size = file.tell()
        return file_size // 4  # Each token is 4 bytes

    def __len__(self):
        return self.num_samples // self.block_size

    def __getitem__(self, idx):
        self.file.seek(idx * 4)  # Each token is 4 bytes
        data = self.file.read(4 * self.block_size)
        token_ids = struct.unpack(f'{self.block_size}I', data)
        return torch.tensor(token_ids, dtype=torch.long)

    def __del__(self):
        self.file.close()

def get_random_batch(data_loader, num_samples=100):
    start_idx = random.randint(0, len(data_loader.dataset) - num_samples)
    batch = torch.stack([data_loader.dataset[i] for i in range(start_idx, start_idx + num_samples)]).to(device)
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
    def __init__(self, head_size, window_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.window_size = window_size
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
        v = self.value(x)  # Value
        
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # Scaled dot-product attention
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Masking for causal (unidirectional) attention
        window_mask = torch.ones(T, T).triu(diagonal=1 + self.window_size).to(x.device)
        wei = wei.masked_fill(window_mask == 1, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)  # Softmax over the last dimension
        wei = self.dropout(wei)
        
        out = wei @ v  # Output of the attention mechanism
        return out

# Multi-head attention combines several heads
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, window_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, window_size) for _ in range(num_heads)])
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
        self.sa = MultiHeadAttention(n_head, head_size, window_size)
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
        q = index.size(dim=1)
        for _ in range(max_new_tokens - q):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :]  # Take the last step
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            index = torch.cat((index, index_next), dim=1)  # Append to the sequence
            yield index_next.item()
        return index

if (__name__ == '__main__'):
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
    print(f'Size of the model: {num_parameters} parameters')

    # Move the model to the appropriate device (GPU/CPU)
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Preprocessing the data and create the tokenized file if that not exist
    create_index_table_if_not_exist(train_file)
    create_index_table_if_not_exist(val_file)

    train_ds = TextDataset(f"{train_file}.pkl", block_size)
    train_sampler = SequentialBatchSampler(train_ds, batch_size)
    train_data_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=4)

    val_ds = TextDataset(f"{val_file}.pkl", block_size)
    val_sampler = SequentialBatchSampler(val_ds, batch_size)
    val_data_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=4)

    # Training loop
    model.train()
    for iter in tqdm(range(max_iters), total=max_iters):
        if iter % eval_iters == 0 and iter > 0:        
            del xb, yb, logits, loss
            torch.cuda.empty_cache()        
            torch.save(model.state_dict(), f'{load_model_name}_and_{iter}.pkl')
            losses = estimate_loss(train_data_loader, val_data_loader)        
            print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
            

        xb, yb = get_random_batch(train_data_loader, batch_size)

        # Forward pass and compute loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())

    # Save the trained model
    torch.save(model.state_dict(), save_model_name)
    print('model saved')
