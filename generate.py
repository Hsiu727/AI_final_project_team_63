import torch
from model import CondTransformer
from miditok import REMI
import pickle

# ---------- 參數 ----------
CKPT_PATH = "best_model.pt"
TOKENIZER_PATH = "tokenizer.json"
EMO2IDX_PATH = "emo2idx.pkl"
GEN2IDX_PATH = "gen2idx.pkl"
OUTPUT_PATH = "generated_output.mid"
MAX_LEN = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 0
TEMPERATURE = 1.2

# ---------- 載入資源 ----------
tokenizer = REMI(params=TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.vocab_size

with open(EMO2IDX_PATH, "rb") as f:
    emo2idx = pickle.load(f)
with open(GEN2IDX_PATH, "rb") as f:
    gen2idx = pickle.load(f)

# ---------- 載入模型 ----------
model = CondTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=512,
    nlayers=12,
    nhead=16,
    emo_num=len(emo2idx),
    gen_num=len(gen2idx),
    max_seq_len=MAX_LEN,
).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# ---------- 輸入條件 ----------
emotion = "angry"
genre = "rock"
if emotion not in emo2idx:
    raise ValueError(f"emotion '{emotion}' 不在 emo2idx!")
if genre not in gen2idx:
    raise ValueError(f"genre '{genre}' 不在 gen2idx!")
emo_idx = emo2idx[emotion]
gen_idx = gen2idx[genre]

# ---------- 生成 tokens ----------
with torch.no_grad():
    seq = [BOS_TOKEN]
    for _ in range(MAX_LEN - 1):
        x = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        emo_t = torch.tensor([emo_idx], dtype=torch.long, device=DEVICE)
        gen_t = torch.tensor([gen_idx], dtype=torch.long, device=DEVICE)
        logits = model(x, emo_t, gen_t)  # [1, L, vocab]
        logits = logits[0, -1] / TEMPERATURE

        # 避免 sampling 到 PAD/BOS
        logits[PAD_TOKEN] = float('-inf')
        logits[BOS_TOKEN] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        seq.append(next_token)
        if next_token == EOS_TOKEN:
            break

print(f"生成完成，token 長度：{len(seq)}")
for tid in seq:
    if tid < 0 or tid >= VOCAB_SIZE:
        print(f"非法 token: {tid}")
    else:
        continue
#         print(tokenizer[tid])

# ---------- tokens 還原成 MIDI ----------
midi = tokenizer(seq)
midi.dump_midi(OUTPUT_PATH)
print(f"已儲存為 {OUTPUT_PATH}")
