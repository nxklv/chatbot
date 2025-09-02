import torch, torch.nn as nn, torch.nn.functional as F, math, argparse, json

# ======== Device ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======== Load dataset ========
with open(r"C:\Users\nakul\smallGPT\dataset.txt","r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# Save vocab for chat.py
with open("vocab.json","w") as f:
    json.dump({"itos": itos, "stoi": stoi}, f)

# ======== Hyperparameters ========
block_size = 64
n_embd = 128
n_head = 4
n_layer = 3
batch_size = 16 
lr = 5e-4      
steps = 10000

# ======== Dataset ========
def get_batch():
    ix = torch.randint(len(data)-block_size-1,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ======== Model ========
class Head(nn.Module):
    """Single attention head"""
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd // n_head, bias=False)
        self.query = nn.Linear(n_embd, n_embd // n_head, bias=False)
        self.value = nn.Linear(n_embd, n_embd // n_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B,T,32)
        q = self.query(x)   # (B,T,32)
        v = self.value(x)   # (B,T,32)
        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v       # (B,T,32)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(n_head)])
        self.proj = nn.Linear((n_embd // n_head) * n_head, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,64)
        out = self.proj(out)  # back to (B,T,64)
        return out

class Block(nn.Module):
    """Transformer block"""
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention()
        self.fc = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())

    def forward(self, x):
        x = x + self.sa(self.ln(x))
        x = x + self.fc(self.ln(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=50, temperature=1.0):
        for _ in range(max_new):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

# ======== Train ========
model = TinyGPT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(steps):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.3f}")

torch.save(model.state_dict(), "gpt.pth")
print("Model saved to gpt.pth")
