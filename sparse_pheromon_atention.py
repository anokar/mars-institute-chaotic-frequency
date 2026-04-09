# ============================================================
# Sparse Pheromon Attention - RESEARCH VERSION (Improved)
# Focus: Stability, Correctness, Experiments
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import requests
import os
from IPython.display import clear_output

# ====================== CONFIG ======================
USE_PHEROMON = True
# ====================== EXPERIMENT MODES ======================
# Phase 1: True  -> sauberes Lernen ohne falsches Memory
# Phase 2: False -> echtes persistentes Gedächtnis testen
RESET_TAU_EACH_BATCH = True

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
BLOCK_SIZE = 64
BATCH_SIZE = 32
MAX_ITERS = 8500
EVAL_INTERVAL = 200

# Pheromon params
K = 4
RHO = 0.085
GAMMA = 2.5
LOCAL_WINDOW = 0
TAU_CLIP = 5.0

LEARNING_RATE = 5e-5

print(f"Mode: {'Pheromon' if USE_PHEROMON else 'Standard'}")

# ====================== ATTENTION ======================
class SparsePheromonAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, tau):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.embed_dim, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        logits = (q @ k.transpose(-2, -1)) * scale

        # ===== pheromon bias =====
        if tau is not None:
            pheromone_bias = GAMMA * torch.log(tau + 1e-8)
            logits = logits + pheromone_bias.unsqueeze(1)

        # ===== mask =====
        # NOTE: Local window can dominate learning.
        # Try reducing or disabling for experiments.
        mask = torch.zeros((B, T, T), dtype=torch.bool, device=x.device)

        # local window
        # Local window (can disable by setting LOCAL_WINDOW = 0)
        if LOCAL_WINDOW > 0:
            for i in range(T):
                start = max(0, i - LOCAL_WINDOW)
                end = min(T, i + LOCAL_WINDOW + 1)
                mask[:, i, start:end] = True

        # top-k per batch (global learned connections)
        # This is where "memory" can emerge
        if tau is not None:
            # Ensure k is not larger than the dimension size
            k_val = min(K, tau.shape[-1])
            topk = torch.topk(tau, k_val, dim=-1).indices
            mask.scatter_(-1, topk, True)

        logits = logits.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(logits, dim=-1)

        # ===== pheromon update =====
        # Key idea: reinforce non-trivial attention patterns
        if tau is not None:
            with torch.no_grad():
                # less self-reinforcing
                baseline = 1.0 / T
                signal = attn_weights.mean(1)
                # amplify stronger-than-average connections
                reinforcement = signal ** 3

                tau = (1 - RHO) * tau + reinforcement

                # normalization + clipping
                tau = tau / (tau.sum(dim=-1, keepdim=True) + 1e-8)
                tau = torch.clamp(tau, max=TAU_CLIP)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)

        return out, tau

# ====================== BLOCK ======================
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = SparsePheromonAttention(embed_dim, num_heads) if USE_PHEROMON else nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x, tau=None):
        if USE_PHEROMON:
            attn_out, tau = self.attn(self.ln1(x), tau)
        else:
            attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))

        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, tau

# ====================== MODEL ======================
class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)

        self.blocks = nn.ModuleList([Block(EMBED_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)])

        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx, tau=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))

        if USE_PHEROMON and tau is None:
            tau = torch.ones((B, T, T), device=idx.device) * 0.01

        for block in self.blocks:
            x, tau = block(x, tau if USE_PHEROMON else None)

        logits = self.head(self.ln_f(x))
        return logits, tau

# ====================== DATA ======================
if not os.path.exists('input.txt'):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open('input.txt', 'w') as f:
        f.write(requests.get(url).text)

text = open('input.txt').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

# ====================== TRAIN ======================
print("\n=== Experiment Mode ===")
print(f"Reset Tau Each Batch: {RESET_TAU_EACH_BATCH}")
print(f"Local Window: {LOCAL_WINDOW}")
print(f"Top-K: {K}\n")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model(vocab_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

losses = []

for step in range(MAX_ITERS):
    xb, yb = get_batch()

    if RESET_TAU_EACH_BATCH:
        tau = None
    # else: keep tau from previous step (persistent memory experiment)

    logits, tau = model(xb, tau)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(loss.item())

    if step % EVAL_INTERVAL == 0:
        clear_output(wait=True)
        print(f"Step {step} | Loss {loss.item():.4f}")

# ====================== VISUALIZE TAU ======================
print("\nVisualizing Tau (memory structure)...")
if USE_PHEROMON and tau is not None:
    plt.imshow(tau[0].detach().cpu())
    plt.title("Tau Matrix")
    plt.colorbar()
    plt.show()

# ====================== GENERATE ======================
context = torch.zeros((1,1), dtype=torch.long, device=device)
for _ in range(300):
    logits, _ = model(context[:, -BLOCK_SIZE:])
    probs = F.softmax(logits[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, 1)
    context = torch.cat([context, next_token], dim=1)

print(''.join([itos[int(i)] for i in context[0]]))
entropy = -(tau * torch.log(tau + 1e-8)).sum(dim=-1).mean()
print("tau entropy:", entropy.item())