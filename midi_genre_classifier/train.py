import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from midi_dataset import PrecomputedMIDIDataset
from midi_dataset import MIDIMultiLabelDataset
from model import MultiTaskMIDIMHAttention
from model import MultiTaskMIDICNN
from model import MultiTaskMIDIConvNeXt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# RTX 4090級參數
BATCH_SIZE = 128
EPOCHS = 30
MAX_LENGTH = 500
LEARNING_RATE = 1.5e-4
ROOT_DIR = 'preprocessed'
CHECKPOINT_PATH = 'midi_multitask_ckpt.pt'

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = PrecomputedMIDIDataset(ROOT_DIR, max_length=MAX_LENGTH)
    num_emotions = len(dataset.emotion2idx)
    num_genres = len(dataset.genre2idx)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # model = MultiTaskMIDIMHAttention(num_emotions, num_genres, max_length=MAX_LENGTH).to(device)
    # model = MultiTaskMIDIConvNeXt(num_emotions, num_genres, max_length=MAX_LENGTH).to(device)
    model = MultiTaskMIDICNN(num_emotions, num_genres, max_length=MAX_LENGTH).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5, verbose=True)

    # --- 新增 resume 機制 ---
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        for x, emo, gen in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x = x.to(device, non_blocking=True)
            emo = emo.to(device, non_blocking=True)
            gen = gen.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                out_emo, out_gen = model(x)
                if epoch == start_epoch and train_loss == 0.0:
                    print("out_emo.shape:", out_emo.shape)
                    print("out_gen.shape:", out_gen.shape)
                    print("emo.shape, gen.shape:", emo.shape, gen.shape)
                    print("emo.dtype, gen.dtype:", emo.dtype, gen.dtype)
                    print("emo.unique:", emo.unique())
                    print("gen.unique:", gen.unique())
                    print("batch pianoroll sum:", x.sum(dim=(1,2)))  # 每首歌的總音量
                loss_emo = criterion(out_emo, emo)
                loss_gen = criterion(out_gen, gen)
                loss = loss_emo + loss_gen
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        # 驗證
        model.eval()
        correct_emo, total_emo = 0, 0
        correct_gen, total_gen = 0, 0
        with torch.no_grad():
            for x, emo, gen in val_loader:
                x = x.to(device, non_blocking=True)
                emo = emo.to(device, non_blocking=True)
                gen = gen.to(device, non_blocking=True)
                with autocast():
                    out_emo, out_gen = model(x)
                pred_emo = out_emo.argmax(dim=1)
                pred_gen = out_gen.argmax(dim=1)
                correct_emo += (pred_emo == emo).sum().item()
                total_emo += emo.size(0)
                correct_gen += (pred_gen == gen).sum().item()
                total_gen += gen.size(0)
        acc_emo = correct_emo / total_emo
        acc_gen = correct_gen / total_gen
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Emotion Acc: {acc_emo:.4f}, Val Genre Acc: {acc_gen:.4f}")

        # --- 每 epoch 自動儲存 checkpoint ---
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'emotion2idx': dataset.emotion2idx,
            'genre2idx': dataset.genre2idx,
            'max_length': MAX_LENGTH
        }, CHECKPOINT_PATH)

    # 訓練結束，儲存最終模型（可選）
    torch.save({
        'model': model.state_dict(),
        'emotion2idx': dataset.emotion2idx,
        'genre2idx': dataset.genre2idx,
        'max_length': MAX_LENGTH
    }, "midi_multitask_cnn_final.pt")
