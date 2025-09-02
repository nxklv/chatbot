import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import re

# --------------------------
# Load vocab
# --------------------------
with open("vocab.json", "r") as f:
    vocab = json.load(f)

itos = {int(i): c for i, c in vocab["itos"].items()}
stoi = vocab["stoi"]
vocab_size = len(stoi)

block_size = 64
n_embd = 128   
n_head = 4     
n_layer = 3         

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Model (must match training)
# --------------------------
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd // n_head, bias=False)
        self.query = nn.Linear(n_embd, n_embd // n_head, bias=False)
        self.value = nn.Linear(n_embd, n_embd // n_head, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(n_head)])
        self.proj = nn.Linear((n_embd // n_head) * n_head, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class Block(nn.Module):
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
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate_until_user(self, idx, temperature=0.8, max_new=200):
        """Generate starting after <a>, stop when <u> is produced"""
        generated = idx.clone()
        for _ in range(max_new):
            logits, _ = self(generated[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_idx], dim=1)

            # decode only the new character
            ch = itos[next_idx.item()]
            if ch == "<u>":  # stop if model starts next user turn
                break
        return generated

# --------------------------
# Load trained model
# --------------------------
model = TinyGPT().to(device)
model.load_state_dict(torch.load("gpt.pth", map_location=device))
model.eval()

# --------------------------
# Encoding helpers
# --------------------------
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# --------------------------
# Clean response
# --------------------------
def clean_response(raw_text, context):
    response = raw_text[len(context):]

    # stop at first <u> (user turn) or newline
    stop_tokens = ["<u>"]
    for tok in stop_tokens:
        if tok in response:
            response = response.split(tok)[0]

    # remove dataset tags
    response = re.sub(r"<[ua]>", "", response)
    return response.strip()

# --------------------------
# Chat loop
# --------------------------
context = "<u>"
print("Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    context += user_input + "<a>"   # user prompt ends with <a>
    x = torch.tensor([encode(context[-block_size:])], dtype=torch.long).to(device)

    y = model.generate_until_user(x, temperature=0.8)
    raw_out = decode(y[0].tolist())
    print("GPT:", clean_response(raw_out, context))

    # Update context with generated assistant response too
    context += clean_response(raw_out, context) + "<u>"
