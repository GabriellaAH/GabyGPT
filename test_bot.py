import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer
from gabyGPT_train import GPTLanguageModel 

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
load_model_name = 'gabyGPT-23.pkl'

print(device)

tokenizer = ByteLevelBPETokenizer("./BPEVocab/vocab.json", "./BPEVocab/merges.txt")
vocab_size = len(tokenizer.get_vocab())

encode = lambda text: tokenizer.encode(text).ids
decode = lambda token_ids: tokenizer.decode(token_ids)

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
    generator = m.generate(context.unsqueeze(0), max_new_tokens=127)
    print('GabyGPT:')
    for next_token in generator:
        generated_char = decode([next_token])
        print(generated_char, end='', flush=True)
    print()  
