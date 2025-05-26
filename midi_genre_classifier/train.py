import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from midi_dataset import MIDIMultiLabelDataset
from model import MultiTaskMIDICNN
from tqdm import tqdm

BATCH_SIZE = 32
EPOCHS = 10
MAX_LENGTH = 500
ROOT_DIR = 'data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

dataset = MIDIMultiLabelDataset(ROOT_DIR, max_length=MAX_LENGTH)
num_emotions = len(dataset.emotion2idx)
num_genres = len(dataset.genre2idx)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = MultiTaskMIDICNN(num_emotions, num_genres, max_length=MAX_LENGTH).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    for x, emo, gen in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x = x.to(device)
        emo = emo.to(device)
        gen = gen.to(device)
        out_emo, out_gen = model(x)
        loss_emo = criterion(out_emo, emo)
        loss_gen = criterion(out_gen, gen)
        loss = loss_emo + loss_gen
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 驗證
    model.eval()
    correct_emo, total_emo = 0, 0
    correct_gen, total_gen = 0, 0
    with torch.no_grad():
        for x, emo, gen in val_loader:
            x = x.to(device)
            emo = emo.to(device)
            gen = gen.to(device)
            out_emo, out_gen = model(x)
            pred_emo = out_emo.argmax(dim=1)
            pred_gen = out_gen.argmax(dim=1)
            correct_emo += (pred_emo == emo).sum().item()
            total_emo += emo.size(0)
            correct_gen += (pred_gen == gen).sum().item()
            total_gen += gen.size(0)
    acc_emo = correct_emo / total_emo
    acc_gen = correct_gen / total_gen
    print(f"Epoch {epoch+1}, Val Emotion Acc: {acc_emo:.4f}, Val Genre Acc: {acc_gen:.4f}")

torch.save({
    'model': model.state_dict(),
    'emotion2idx': dataset.emotion2idx,
    'genre2idx': dataset.genre2idx
}, "midi_multitask_cnn.pt")
