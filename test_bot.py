import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from tokenizers import ByteLevelBPETokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

batch_size = 16
block_size = 128
max_iters = 30000
learning_rate = 4e-4
eval_iters = 500
n_embd = 1140
n_head = 32
n_layer = 36
dropout = 0.2
load_model_name = 'gabyGPT-00.pkl'

print(device)

tokenizer = ByteLevelBPETokenizer("./BPEVocab/vocab.json", "./BPEVocab/merges.txt")
vocab_size = len(tokenizer.get_vocab())

encode = lambda text: tokenizer.encode(text).ids
decode = lambda token_ids: tokenizer.decode(token_ids)


class Head(nn.Module):
    """ one head of self-attention """

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
    

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
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
            yield index_next.item()
        return index

model = GPTLanguageModel(vocab_size)
print('loading model parameters...')
model.load_state_dict(torch.load(load_model_name))
# with open(load_model_name, 'rb') as f:
#     model = pickle.load(f)
print('loaded successfully!')
m = model.to(device)



while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generator = m.generate(context.unsqueeze(0), max_new_tokens=150)
    print('GabyGPT:')
    for next_token in generator:
        generated_char = decode([next_token])
        print(generated_char, end='', flush=True)
    print()  
