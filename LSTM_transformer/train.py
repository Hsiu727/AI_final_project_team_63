import os
import pickle
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CondTransformer, CondLSTM
from dataset import FullBandMusicDataset
from collate_fn import music_collate_fn
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ----- 參數設定與命令列參數解析 -----
parser = argparse.ArgumentParser(description="Train a conditional music generation model")
parser.add_argument("--model", choices=["transformer", "lstm"], default="transformer",
                    help="Model type to train")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--max_len", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--val_ratio", type=float, default=0.2)
parser.add_argument("--d_model", type=int, default=256)
# LSTM-specific args
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.1)
args = parser.parse_args()

# 常數
DATASET_PATH  = "dataset_fullband.pkl"
EMO2IDX_PATH  = "emo2idx.pkl"
GEN2IDX_PATH  = "gen2idx.pkl"
TOKENIZER_PATH= "tokenizer.json"
PAD_TOKEN     = 0
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH     = f"best_{args.model}.pt"

# ----- 載入 tokenizer -----
from miditok import REMI
tokenizer = REMI(params=TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.vocab_size

# ----- 載入 & 切分資料 -----
with open(DATASET_PATH, "rb") as f:
    all_data = pickle.load(f)
train_data, val_data = train_test_split(all_data, test_size=args.val_ratio, random_state=42)
with open("train_split.pkl", "wb") as f: pickle.dump(train_data, f)
with open("val_split.pkl", "wb") as f: pickle.dump(val_data, f)

# ----- 建立 Dataset 與 DataLoader -----
emo2idx = pickle.load(open(EMO2IDX_PATH, "rb"))
gen2idx = pickle.load(open(GEN2IDX_PATH, "rb"))
train_dataset = FullBandMusicDataset("train_split.pkl", EMO2IDX_PATH, GEN2IDX_PATH,
                                    max_len=args.max_len, pad_token=PAD_TOKEN)
val_dataset   = FullBandMusicDataset("val_split.pkl",   EMO2IDX_PATH, GEN2IDX_PATH,
                                    max_len=args.max_len, pad_token=PAD_TOKEN)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, collate_fn=music_collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                          drop_last=False, collate_fn=music_collate_fn)

print(f"Train dataset size: {len(train_dataset)} | batches: {len(train_loader)}")
print(f"Val   dataset size: {len(val_dataset)}   | batches: {len(val_loader)}")

# ----- 建立 Model -----
if args.model == "transformer":
    model = CondTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        nlayers=6,
        nhead=8,
        emo_num=len(emo2idx),
        gen_num=len(gen2idx),
        max_seq_len=args.max_len
    )
else:
    model = CondLSTM(
        vocab_size=VOCAB_SIZE,
        emo_num=len(emo2idx),
        gen_num=len(gen2idx),
        event_type_num=None,  # adjust if needed
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_seq_len=args.max_len,
        dropout=args.dropout
    )
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

# ----- accuracy 計算 -----
def compute_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    mask = (targets != PAD_TOKEN)
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0

# ----- 驗證函數 -----
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, total_acc, total_tokens = 0, 0, 0
    for tokens, emos, gens, lengths in loader:
        tokens = tokens.to(DEVICE); emos = emos.to(DEVICE); gens = gens.to(DEVICE)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        logits = model(inputs, emos, gens)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        total_loss += loss.item() * targets.ne(PAD_TOKEN).sum().item()
        acc = compute_accuracy(logits, targets)
        total_acc += acc * targets.ne(PAD_TOKEN).sum().item()
        total_tokens += targets.ne(PAD_TOKEN).sum().item()
    return total_loss/total_tokens, total_acc/total_tokens

# ----- 訓練迴圈 -----
best_val_loss = float('inf')
for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss, running_acc, running_tokens = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
    for tokens, emos, gens, lengths in pbar:
        tokens = tokens.to(DEVICE); emos = emos.to(DEVICE); gens = gens.to(DEVICE)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        logits = model(inputs, emos, gens)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = targets.ne(PAD_TOKEN).sum().item()
        running_loss += loss.item() * bs
        acc = compute_accuracy(logits, targets)
        running_acc += acc * bs
        running_tokens += bs
        pbar.set_postfix(loss=loss.item(), acc=acc)
    train_loss = running_loss/running_tokens
    train_acc  = running_acc/running_tokens
    val_loss, val_acc = evaluate(val_loader)
    print(f"[Epoch {epoch}] train_loss: {train_loss:.4f} acc: {train_acc:.4f} "
          f"| val_loss: {val_loss:.4f} acc: {val_acc:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"New best model saved to {CKPT_PATH}")
