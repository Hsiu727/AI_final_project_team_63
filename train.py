import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CondTransformer
from dataset import FullBandMusicDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ----- 參數設定 -----
DATASET_PATH = "dataset_fullband.pkl"
EMO2IDX_PATH = "emo2idx.pkl"
GEN2IDX_PATH = "gen2idx.pkl"
TOKENIZER_PATH = "tokenizer.json"
BATCH_SIZE = 64
EPOCHS = 30
MAX_LEN = 512
PAD_TOKEN = 0
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_model.pt"
VAL_RATIO = 0.2

# ----- 載入 tokenizer -----
from miditok import REMI
tokenizer = REMI(params=TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.vocab_size

# ----- 載入 & 切分資料（必須寫檔案！） -----
with open(DATASET_PATH, "rb") as f:
    all_data = pickle.load(f)

train_data, val_data = train_test_split(all_data, test_size=VAL_RATIO, random_state=42)

# 把切分後的資料各自寫入新檔
with open("train_split.pkl", "wb") as f:
    pickle.dump(train_data, f)
with open("val_split.pkl", "wb") as f:
    pickle.dump(val_data, f)

# ----- 資料集與 DataLoader -----
from dataset import FullBandMusicDataset

train_dataset = FullBandMusicDataset("train_split.pkl", EMO2IDX_PATH, GEN2IDX_PATH, max_len=MAX_LEN, pad_token=PAD_TOKEN)
val_dataset = FullBandMusicDataset("val_split.pkl", EMO2IDX_PATH, GEN2IDX_PATH, max_len=MAX_LEN, pad_token=PAD_TOKEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Train loader batch count: {len(train_loader)}")
print(f"Val loader batch count: {len(val_loader)}")

# ----- 模型 -----
with open(EMO2IDX_PATH, "rb") as f:
    emo2idx = pickle.load(f)
with open(GEN2IDX_PATH, "rb") as f:
    gen2idx = pickle.load(f)

model = CondTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=256,
    nlayers=6,
    nhead=8,
    emo_num=len(emo2idx),
    gen_num=len(gen2idx),
    max_seq_len=MAX_LEN,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)


# ----- accuracy 計算函數 -----
def compute_accuracy(logits, targets, pad_token):
    """
    logits: [batch, seq_len-1, vocab_size]
    targets: [batch, seq_len-1]
    """
    preds = torch.argmax(logits, dim=-1)           # [batch, seq_len-1]
    mask = (targets != pad_token)                   # [batch, seq_len-1]
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    acc = correct / total if total > 0 else 0
    return acc

# ----- 驗證函數 -----
@torch.no_grad()
def evaluate(model, val_loader, criterion, device, pad_token):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_count = 0
    for tokens, emos, gens, lengths in val_loader:
        tokens = tokens.to(device)
        emos = emos.to(device)
        gens = gens.to(device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits = model(inputs, emos, gens)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        total_loss += loss.item()
        acc = compute_accuracy(logits, targets, pad_token)
        batch_size = tokens.size(0)
        total_acc += acc * batch_size
        total_count += batch_size
    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / total_count
    return avg_loss, avg_acc

# ----- 訓練迴圈 -----
best_val_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_acc = 0
    total_loss = 0
    total_count = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", dynamic_ncols=True)
    for tokens, emos, gens, lengths in pbar:
        tokens = tokens.to(DEVICE)
        emos = emos.to(DEVICE)
        gens = gens.to(DEVICE)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits = model(inputs, emos, gens)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc = compute_accuracy(logits, targets, PAD_TOKEN)
        batch_size = tokens.size(0)
        total_acc += acc * batch_size
        total_count += batch_size
        pbar.set_postfix(loss=loss.item(), acc=acc)

    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc = total_acc / total_count
    print(f"[Epoch {epoch}] train loss: {avg_train_loss:.4f} | train acc: {avg_train_acc:.4f}")

    # validation
    val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE, PAD_TOKEN)
    print(f"[Epoch {epoch}] val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")

    # best model by val loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"  (Saved new best model to {CKPT_PATH})")
