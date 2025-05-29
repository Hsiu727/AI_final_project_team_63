import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CondTransformer
from dataset import FullBandMusicDataset
import pickle
from tqdm import tqdm 

# ----- 參數設定 -----
DATASET_PATH = "dataset_fullband.pkl"
EMO2IDX_PATH = "emo2idx.pkl"
GEN2IDX_PATH = "gen2idx.pkl"
TOKENIZER_PATH = "tokenizer.json"
BATCH_SIZE = 16
EPOCHS = 5
MAX_LEN = 512
PAD_TOKEN = 0
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_model.pt"

# ----- 載入 tokenizer -----
from miditok import REMI
tokenizer = REMI(params=TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.vocab_size

# ----- 資料集與 DataLoader -----
train_dataset = FullBandMusicDataset(DATASET_PATH, EMO2IDX_PATH, GEN2IDX_PATH, max_len=MAX_LEN, pad_token=PAD_TOKEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Train loader batch count: {len(train_loader)}")

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

# ----- 訓練迴圈 -----
best_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    # 加入 tqdm 進度條
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
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch}] train loss: {avg_loss:.4f}")

    # 儲存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"  (Saved new best model to {CKPT_PATH})")
