# train.py
import os
import pickle
import torch
from torch.utils.data import DataLoader
from dataset import MusicDataset
from collate_fn import music_collate_fn
from model import CondTransformer
from tqdm import tqdm
from miditok import REMI

# ------------------
# Config & Hyperparams
# ------------------
DATA_PKL   = 'dataset.pkl'
EMO_PKL    = 'emo2idx.pkl'
GEN_PKL    = 'gen2idx.pkl'
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 16
MAX_LEN    = 512
LR         = 1e-3
EPOCHS     = 10
TOKENIZER_JSON = "tokenizer.json"
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------
# Load Data & Vocab
# ------------------
with open(DATA_PKL, 'rb') as f:
    data = pickle.load(f)
with open(EMO_PKL, 'rb') as f:
    emo2idx = pickle.load(f)
with open(GEN_PKL, 'rb') as f:
    gen2idx = pickle.load(f)

# Determine vocab size from tokenizer saved params if available
# For simplicity, user should set VOCAB_SIZE manually or load from tokenizer
tokenizer = REMI(params=TOKENIZER_JSON) if TOKENIZER_JSON else REMI()
VOCAB_SIZE = tokenizer.vocab_size if 'tokenizer' in globals() else 512  # update accordingly

# ------------------
# Dataset & Dataloader
# ------------------
train_dataset = MusicDataset(data, emo2idx, gen2idx, max_len=MAX_LEN, pad_token=0)
train_loader  = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=music_collate_fn
)

# ------------------
# Model, Optimizer, Loss
# ------------------
model = CondTransformer(
    vocab_size=VOCAB_SIZE,
    emo_num=len(emo2idx),
    gen_num=len(gen2idx),
    d_model=256,
    nhead=8,
    nlayers=4,
    max_seq_len=MAX_LEN
).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = torch.nn.CrossEntropyLoss(ignore_index=0)

# ------------------
# Training Loop
# ------------------
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for tokens, attn_mask, emo, gen, lengths in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        tokens = tokens.to(DEVICE)           # [B, L]
        emo    = emo.to(DEVICE)              # [B]
        gen    = gen.to(DEVICE)              # [B]
        # Shift inputs and targets for auto-regressive training
        inputs  = tokens[:, :-1]
        targets = tokens[:, 1:]
        # Forward
        logits = model(inputs, emo, gen)    # [B, L-1, vocab_size]
        # Compute loss: flatten
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f}")
    # Save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'epoch{epoch}.pt')
    torch.save(model.state_dict(), ckpt_path)

print("Training complete!")