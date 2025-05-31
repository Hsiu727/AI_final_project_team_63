import os
import csv
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CondTransformer
from dataset import FullBandMusicDataset
from sklearn.model_selection import train_test_split
import time

# ---- Default/fixed values ----
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_NLAYERS = 8
EPOCHS = 10  # For demo, you can set higher
MAX_LEN = 512
PAD_TOKEN = 0
VAL_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "dataset_fullband.pkl"
EMO2IDX_PATH = "emo2idx.pkl"
GEN2IDX_PATH = "gen2idx.pkl"
TOKENIZER_PATH = "tokenizer.json"
EPOCH_TIME_LIMIT = 600  # seconds (10 mins) per epoch, change as needed

from miditok import REMI
tokenizer = REMI(params=TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.vocab_size

# Load and split dataset (only once)
with open(DATASET_PATH, "rb") as f:
    all_data = pickle.load(f)
train_data, val_data = train_test_split(all_data, test_size=VAL_RATIO, random_state=42)
with open("train_split.pkl", "wb") as f:
    pickle.dump(train_data, f)
with open("val_split.pkl", "wb") as f:
    pickle.dump(val_data, f)

# Prepare results CSV
RESULTS_CSV = "controlled_experiment_results.csv"
with open(RESULTS_CSV, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["test_type", "value", "final_train_loss", "final_train_acc", "final_val_loss", "final_val_acc", "status"])

def compute_accuracy(logits, targets, pad_token):
    preds = torch.argmax(logits, dim=-1)
    mask = (targets != pad_token)
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0

@torch.no_grad()
def evaluate(model, loader, criterion, device, pad_token):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_count = 0
    for tokens, emos, gens, lengths in loader:
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
    return total_loss / len(loader), total_acc / total_count

def safe_train(
    model, optimizer, criterion, train_loader, val_loader, epochs, test_type, value,
    device, pad_token, results_csv, epoch_time_limit
):
    status = "OK"
    final_train_loss = final_train_acc = final_val_loss = final_val_acc = None
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            total_acc = 0
            total_count = 0
            start_time = time.time()
            pbar = tqdm(train_loader, desc=f"{test_type}={value} | Epoch {epoch}/{epochs}", dynamic_ncols=True, leave=False)
            for tokens, emos, gens, lengths in pbar:
                tokens = tokens.to(device)
                emos = emos.to(device)
                gens = gens.to(device)
                inputs = tokens[:, :-1]
                targets = tokens[:, 1:]
                try:
                    logits = model(inputs, emos, gens)
                    loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        status = "ERROR: GPU OOM"
                        print(f"\n[!] GPU OOM detected for {test_type}={value} (batch size: {train_loader.batch_size})")
                        raise e
                    else:
                        raise e
                total_loss += loss.item()
                acc = compute_accuracy(logits, targets, pad_token)
                batch_size = tokens.size(0)
                total_acc += acc * batch_size
                total_count += batch_size
                pbar.set_postfix(loss=loss.item(), acc=acc)
            elapsed = time.time() - start_time
            if elapsed > epoch_time_limit:
                status = f"ERROR: Training time exceeded ({elapsed:.1f}s)"
                print(f"\n[!] Training time exceeded for {test_type}={value}: {elapsed:.1f}s")
                break  # Stop training this experiment if too slow

        if status == "OK":
            final_train_loss, final_train_acc = evaluate(model, train_loader, criterion, device, pad_token)
            final_val_loss, final_val_acc = evaluate(model, val_loader, criterion, device, pad_token)
    except RuntimeError as e:
        if status == "ERROR: GPU OOM":
            # Save error to csv and return
            with open(results_csv, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([test_type, value, None, None, None, None, status])
            return
        else:
            raise e

    with open(results_csv, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([test_type, value, final_train_loss, final_train_acc, final_val_loss, final_val_acc, status])

# ---- Learning Rate experiment ----
for lr in [1e-3, 5e-4, 1e-4, 5e-5]:
    print(f"\n==== Testing Learning Rate: {lr} ====")
    train_dataset = FullBandMusicDataset("train_split.pkl", EMO2IDX_PATH, GEN2IDX_PATH, max_len=MAX_LEN, pad_token=PAD_TOKEN)
    val_dataset = FullBandMusicDataset("val_split.pkl", EMO2IDX_PATH, GEN2IDX_PATH, max_len=MAX_LEN, pad_token=PAD_TOKEN)
    train_loader = DataLoader(train_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    with open(EMO2IDX_PATH, "rb") as f:
        emo2idx = pickle.load(f)
    with open(GEN2IDX_PATH, "rb") as f:
        gen2idx = pickle.load(f)
    model = CondTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        nlayers=DEFAULT_NLAYERS,
        nhead=8,
        emo_num=len(emo2idx),
        gen_num=len(gen2idx),
        max_seq_len=MAX_LEN,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    safe_train(
        model, optimizer, criterion, train_loader, val_loader, EPOCHS, "learning_rate", lr,
        DEVICE, PAD_TOKEN, RESULTS_CSV, EPOCH_TIME_LIMIT
    )

# ---- Model Depth experiment ----
for nlayers in [6, 8, 12]:
    print(f"\n==== Testing Model Depth (nlayers): {nlayers} ====")
    train_dataset = FullBandMusicDataset("train_split.pkl", EMO2IDX_PATH, GEN2IDX_PATH, max_len=MAX_LEN, pad_token=PAD_TOKEN)
    val_dataset = FullBandMusicDataset("val_split.pkl", EMO2IDX_PATH, GEN2IDX_PATH, max_len=MAX_LEN, pad_token=PAD_TOKEN)
    train_loader = DataLoader(train_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    with open(EMO2IDX_PATH, "rb") as f:
        emo2idx = pickle.load(f)
    with open(GEN2IDX_PATH, "rb") as f:
        gen2idx = pickle.load(f)
    model = CondTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        nlayers=nlayers,
        nhead=8,
        emo_num=len(emo2idx),
        gen_num=len(gen2idx),
        max_seq_len=MAX_LEN,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    safe_train(
        model, optimizer, criterion, train_loader, val_loader, EPOCHS, "nlayers", nlayers,
        DEVICE, PAD_TOKEN, RESULTS_CSV, EPOCH_TIME_LIMIT
    )

# ---- Batch Size experiment (last) ----
for bs in [32, 64, 128]:
    print(f"\n==== Testing Batch Size: {bs} ====")
    train_dataset = FullBandMusicDataset("train_split.pkl", EMO2IDX_PATH, GEN2IDX_PATH, max_len=MAX_LEN, pad_token=PAD_TOKEN)
    val_dataset = FullBandMusicDataset("val_split.pkl", EMO2IDX_PATH, GEN2IDX_PATH, max_len=MAX_LEN, pad_token=PAD_TOKEN)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    with open(EMO2IDX_PATH, "rb") as f:
        emo2idx = pickle.load(f)
    with open(GEN2IDX_PATH, "rb") as f:
        gen2idx = pickle.load(f)
    model = CondTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        nlayers=DEFAULT_NLAYERS,
        nhead=8,
        emo_num=len(emo2idx),
        gen_num=len(gen2idx),
        max_seq_len=MAX_LEN,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    safe_train(
        model, optimizer, criterion, train_loader, val_loader, EPOCHS, "batch_size", bs,
        DEVICE, PAD_TOKEN, RESULTS_CSV, EPOCH_TIME_LIMIT
    )

print("\nAll experiments finished! See 'controlled_experiment_results.csv' for summary.")
