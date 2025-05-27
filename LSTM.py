import os
import re
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pretty_midi
from mido import MidiFile, MidiTrack, Message

DATA_DIR    = "data_"
EPOCHS      = 10
BATCH_SIZE  = 16
SEQLEN      = 200
TOP_K       = 4


# --------------------
# Helpers for label parsing
# --------------------
def parse_emo_gen(path):
    """
    從檔名解析 emotion 和 genre，假設格式包含 _<emo>_<gen>_
    """
    fn = os.path.basename(path)
    m = re.match(r'.*_(?P<emo>[A-Za-z]+)_(?P<gen>[A-Za-z]+)_', fn)
    if not m:
        raise ValueError(f"Filename '{fn}' 不符合 _emo_gen_ 格式")
    return m.group('emo'), m.group('gen')

def collect_labels(data_dir):
    """
    掃描資料夾，回傳所有 midi 路徑，以及 emotion, genre, instrument program 的 label 列表
    """
    midi_paths = []
    for f in os.listdir(data_dir):
        if f.lower().endswith(('.mid', '.midi')):
            midi_paths.append(os.path.join(data_dir, f))

    emos = set()
    gens = set()
    insts = set()
    for path in midi_paths:
        emo, gen = parse_emo_gen(path)
        emos.add(emo)
        gens.add(gen)
        pm = pretty_midi.PrettyMIDI(path)
        for ins in pm.instruments:
            if not ins.is_drum:
                insts.add(ins.program)
    return midi_paths, sorted(emos), sorted(gens), sorted(insts)

# --------------------
# Dataset
# --------------------
class MusicDataset(Dataset):
    def __init__(self, paths, emo2idx, gen2idx, inst2idx, seq_len):
        self.paths    = paths
        self.emo2idx  = emo2idx
        self.gen2idx  = gen2idx
        self.inst2idx = inst2idx
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # parse emotion & genre
        emo, gen = parse_emo_gen(path)
        e_idx = self.emo2idx[emo]
        g_idx = self.gen2idx[gen]

        # read MIDI, collect all non-drum pitches sequentially
        pm = pretty_midi.PrettyMIDI(path)
        pitches = []
        for ins in pm.instruments:
            if not ins.is_drum:
                pitches += [n.pitch for n in ins.notes]
        # pad/truncate to length+1
        L = self.seq_len + 1
        if len(pitches) < L:
            pitches += [0] * (L - len(pitches))
        else:
            pitches = pitches[:L]
        notes_in  = torch.tensor(pitches[:-1], dtype=torch.long)
        notes_tgt = torch.tensor(pitches[1:],  dtype=torch.long)

        # build multi-hot instrument label
        mhot = torch.zeros(len(self.inst2idx), dtype=torch.float)
        for ins in pm.instruments:
            if not ins.is_drum:
                idx_p = self.inst2idx.get(ins.program)
                if idx_p is not None:
                    mhot[idx_p] = 1.0

        return notes_in, notes_tgt, e_idx, g_idx, mhot

# --------------------
# Model
# --------------------
class MusicLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 cond_dim,
                 input_dim,
                 hidden_dim,
                 num_layers,
                 num_instruments):
        super().__init__()
        self.note_emb  = nn.Embedding(vocab_size, input_dim)
        self.cond_proj = nn.Linear(cond_dim, input_dim)
        self.lstm      = nn.LSTM(input_dim * 2,
                                 hidden_dim,
                                 num_layers,
                                 batch_first=True)
        self.fc_note   = nn.Linear(hidden_dim, vocab_size)
        # instrument multi-label head
        self.inst_head = nn.Linear(hidden_dim, num_instruments)

    def forward(self, notes, cond):
        # notes: (B, L), cond: (B, cond_dim)
        B, L = notes.size()
        # project cond -> (B, input_dim) -> expand to (B, L, input_dim)
        c = self.cond_proj(cond).unsqueeze(1).expand(B, L, -1)
        # embed notes -> (B, L, input_dim)
        n = self.note_emb(notes)
        # concat -> (B, L, 2*input_dim)
        inp = torch.cat([n, c], dim=-1)
        # LSTM
        out, (hn, cn) = self.lstm(inp)  # out: (B, L, hidden_dim)
        # note prediction
        note_logits = self.fc_note(out.reshape(B * L, -1))  # (B*L, vocab_size)
        # instrument multi-label: use mean-pooled hidden
        pooled = out.mean(dim=1)   # (B, hidden_dim)
        inst_logits = self.inst_head(pooled)  # (B, num_instruments)
        return note_logits, inst_logits

# --------------------
# Generation: multi-track
# --------------------
def generate_multi_track(model, cond, seq_len, top_k, inst2idx, idx2inst, device):
    model.eval()
    # 1) predict instrument set
    dummy = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    with torch.no_grad():
        _, inst_logits = model(dummy, cond)
        probs = torch.sigmoid(inst_logits[0])
        # select top_k instruments
        top_idxs = probs.topk(top_k).indices.tolist()

    # 2) generate one track per instrument
    mid = MidiFile()
    for prog in top_idxs:
        track = MidiTrack()
        track.append(Message('program_change', program=idx2inst[prog], time=0))
        mid.tracks.append(track)
        prev = torch.tensor([[0]], dtype=torch.long, device=device)
        hidden = None
        for _ in range(seq_len):
            nl, _ = model(prev, cond)
            prob = F.softmax(nl[-1], dim=0)
            note = torch.multinomial(prob, 1).item()
            track.append(Message('note_on',  note=note, velocity=64, time=120))
            track.append(Message('note_off', note=note, velocity=64, time=120))
            prev = torch.tensor([[note]], dtype=torch.long, device=device)
    return mid

# --------------------
# Training + Interactive
# --------------------
def train_and_generate(data_dir, epochs, batch_size, seq_len, top_k):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # collect labels & paths
    paths, emos, gens, insts = collect_labels(data_dir)
    emo2idx = {e:i for i,e in enumerate(emos)}
    gen2idx = {g:i for i,g in enumerate(gens)}
    inst2idx= {p:i for i,p in enumerate(insts)}
    idx2inst= {i:p for p,i in inst2idx.items()}

    # dataset & loader
    ds = MusicDataset(paths, emo2idx, gen2idx, inst2idx, seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    # model setup
    vocab_size    = 128
    cond_dim      = len(emos) + len(gens)
    input_dim     = 128
    hidden_dim    = 256
    num_layers    = 2
    num_instruments = len(insts)

    model = MusicLSTM(vocab_size, cond_dim,
                      input_dim, hidden_dim, num_layers,
                      num_instruments).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_note = nn.CrossEntropyLoss()
    loss_inst = nn.BCEWithLogitsLoss()
    lambda_inst = 0.5

    # training loop
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for notes_in, notes_tgt, e_idx, g_idx, mhot in tqdm(dl, desc=f"Epoch {ep+1}/{epochs}"):
            # build cond vector
            be = F.one_hot(e_idx, len(emos))
            bg = F.one_hot(g_idx, len(gens))
            cond = torch.cat([be, bg], dim=1).float().to(device)

            notes_in  = notes_in.to(device)
            notes_tgt = notes_tgt.view(-1).to(device)
            mhot      = mhot.to(device)

            optimizer.zero_grad()
            note_logits, inst_logits = model(notes_in, cond)
            ln = loss_note(note_logits, notes_tgt)
            li = loss_inst(inst_logits, mhot)
            loss = ln + lambda_inst * li
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1}, avg loss = {total_loss/len(dl):.4f}")

    # interactive generation
    print("\nAvailable emotions:", emos)
    emo = input("Choose emotion: ").strip()
    print("Available genres:  ", gens)
    gen = input("Choose genre:   ").strip()
    if emo not in emo2idx or gen not in gen2idx:
        print("Invalid choice.")
        return

    # prepare cond for generation
    be = F.one_hot(torch.tensor([emo2idx[emo]]), len(emos))
    bg = F.one_hot(torch.tensor([gen2idx[gen]]), len(gens))
    cond = torch.cat([be, bg], dim=1).float().to(device)

    mid = generate_multi_track(model, cond, seq_len, top_k,
                               inst2idx, idx2inst, device)
    out_path = f"generated_{emo}_{gen}.mid"
    mid.save(out_path)
    print(f"Saved: {out_path}")

# --------------------
# Main entry
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM on MIDI with multi-label instrument prediction and generate multi-track music."
    )
    train_and_generate(DATA_DIR,EPOCHS,BATCH_SIZE,SEQLEN ,TOP_K)
