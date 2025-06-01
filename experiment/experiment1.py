import os
import csv
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CondTransformer  # Ensure this has correct options for model_type!
from dataset import FullBandMusicDataset
from sklearn.model_selection import train_test_split
import numpy as np

# ---- Settings ----
EPOCHS = 20
BATCH_SIZE = 64
MAX_LEN = 512
PAD_TOKEN = 0
VAL_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "dataset_fullband.pkl"
EMO2IDX_PATH = "emo2idx.pkl"
GEN2IDX_PATH = "gen2idx.pkl"
TOKENIZER_PATH = "tokenizer.json"

# -- For checkpoint & CSV --
CKPT_DIR = "exp1_ckpts"
CSV_FILE = "exp01_clean_vs_noisy_results.csv"
os.makedirs(CKPT_DIR, exist_ok=True)

from miditok import REMI
tokenizer = REMI(params=TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.vocab_size

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

def add_gaussian_noise(data, stddev=0.1):
    noisy_data = []
    for sample in data:
        tokens = np.array(sample['tokens'])
        tokens = tokens + np.random.normal(0, stddev, size=tokens.shape)
        tokens = np.clip(tokens, 0, VOCAB_SIZE-1).astype(int)
        new_sample = sample.copy()
        new_sample['tokens'] = tokens.tolist()
        noisy_data.append(new_sample)
    return noisy_data

def get_model(model_type, d_model=256, nlayers=8, nhead=8, max_seq_len=MAX_LEN):
    if model_type == "Base":
        return CondTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            nlayers=nlayers,
            nhead=nhead,
            emo_num=len(emo2idx),
            gen_num=len(gen2idx),
            max_seq_len=max_seq_len,
            attn_dropout=0.0,
            use_layernorm=False
        ).to(DEVICE)
    elif model_type == "AttentionDropout":
        return CondTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            nlayers=nlayers,
            nhead=nhead,
            emo_num=len(emo2idx),
            gen_num=len(gen2idx),
            max_seq_len=max_seq_len,
            attn_dropout=0.3,
            use_layernorm=False
        ).to(DEVICE)
    elif model_type == "LayerNorm":
        return CondTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            nlayers=nlayers,
            nhead=nhead,
            emo_num=len(emo2idx),
            gen_num=len(gen2idx),
            max_seq_len=max_seq_len,
            attn_dropout=0.0,
            use_layernorm=True
        ).to(DEVICE)
    else:
        raise ValueError("Unknown model type.")

def train_and_eval(
    dataset_label, model_type, train_data, val_data,
    resume=False, checkpoint_dir=CKPT_DIR, 
    csv_file=CSV_FILE, 
    max_epochs=EPOCHS
):
    import pickle
    import tempfile

    def save_temp_pickle(obj):
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        pickle.dump(obj, f)
        f.close()
        return f.name

    # --- unique checkpoint file for every dataset/model combination ---
    checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_label}_{model_type}_ckpt.pt")

    train_data_path = save_temp_pickle(train_data)
    val_data_path = save_temp_pickle(val_data)
    train_dataset = FullBandMusicDataset(
        train_data_path, EMO2IDX_PATH, GEN2IDX_PATH, max_len=MAX_LEN, pad_token=PAD_TOKEN
    )
    val_dataset = FullBandMusicDataset(
        val_data_path, EMO2IDX_PATH, GEN2IDX_PATH, max_len=MAX_LEN, pad_token=PAD_TOKEN
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = get_model(model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    start_epoch = 1
    best_val_loss = float("inf")
    train_history, val_history = [], []

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        train_history = checkpoint.get("train_history", [])
        val_history = checkpoint.get("val_history", [])
        print(f"[Resume] {dataset_label} {model_type} at epoch {start_epoch}")

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        total_loss = 0
        total_acc = 0
        total_count = 0
        pbar = tqdm(train_loader, desc=f"{dataset_label} {model_type} Epoch {epoch}/{max_epochs}", dynamic_ncols=True, leave=False)
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
            batch_size_real = tokens.size(0)
            total_acc += acc * batch_size_real
            total_count += batch_size_real
            pbar.set_postfix(loss=loss.item(), acc=acc)

        train_loss = total_loss / len(train_loader)
        train_acc = total_acc / total_count

        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE, PAD_TOKEN)
        train_history.append((epoch, train_loss, train_acc))
        val_history.append((epoch, val_loss, val_acc))

        # CSV logging (append mode)
        with open(csv_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                dataset_label, model_type, epoch, train_loss, train_acc, val_loss, val_acc
            ])
        print(f"[{dataset_label}][{model_type}] Epoch {epoch} - train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        # Save checkpoint after each epoch (unique per model/data combo)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": min(val_loss, best_val_loss),
            "train_history": train_history,
            "val_history": val_history,
        }, checkpoint_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # Clean up temp files
    os.remove(train_data_path)
    os.remove(val_data_path)

    # Return best val acc and loss for summary
    best_epoch, best_val_loss, best_val_acc = max(val_history, key=lambda x: x[2])
    return best_val_loss, best_val_acc, best_epoch

# -- Main script --
if __name__ == "__main__":
    # Prepare meta info & data
    with open(EMO2IDX_PATH, "rb") as f:
        emo2idx = pickle.load(f)
    with open(GEN2IDX_PATH, "rb") as f:
        gen2idx = pickle.load(f)
    with open(DATASET_PATH, "rb") as f:
        all_data = pickle.load(f)

    # Data splits
    train_data_clean, val_data_clean = train_test_split(all_data, test_size=VAL_RATIO, random_state=42)
    train_data_noisy, val_data_noisy = train_test_split(add_gaussian_noise(all_data), test_size=VAL_RATIO, random_state=42)

    model_types = ["Base", "AttentionDropout", "LayerNorm"]

    # Write header if not exists
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        with open(CSV_FILE, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["dataset","model_type","epoch","train_loss","train_acc","val_loss","val_acc"])

    for dataset_label, (train_d, val_d) in [
        ("clean", (train_data_clean, val_data_clean)),
        ("noisy", (train_data_noisy, val_data_noisy))
    ]:
        for model_type in model_types:
            train_and_eval(
                dataset_label, model_type, train_d, val_d,
                resume=True, checkpoint_dir=CKPT_DIR, csv_file=CSV_FILE, max_epochs=EPOCHS
            )
