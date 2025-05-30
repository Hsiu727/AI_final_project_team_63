import torch
import pickle
from model import CondTransformer, CondLSTM
from miditok import REMI

# ==== 手動在這裡設定參數 ====
MODEL_TYPE = "lstm"              # "transformer" 或 "lstm"
CKPT_PATH = "best_model.pt"      # 欲載入的模型權重檔
EMOTION = "angry"                # 產生的 emotion（需在 emo2idx.pkl 裡）
GENRE = "classical"              # 產生的 genre（需在 gen2idx.pkl 裡）
OUTPUT_PATH = "generated.mid"    # 產生的 MIDI 檔案名稱
MAX_LEN = 1024                   # 最大生成長度
TEMPERATURE = 1.2                # softmax sampling 溫度
TOKENIZER_PATH = "tokenizer.json"
EMO2IDX_PATH = "emo2idx.pkl"
GEN2IDX_PATH = "gen2idx.pkl"
# ===========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 讀取 label dicts
with open(EMO2IDX_PATH, "rb") as f:
    emo2idx = pickle.load(f)
with open(GEN2IDX_PATH, "rb") as f:
    gen2idx = pickle.load(f)

if EMOTION not in emo2idx:
    raise ValueError(f"Emotion '{EMOTION}' not found! 可選: {list(emo2idx.keys())}")
if GENRE not in gen2idx:
    raise ValueError(f"Genre '{GENRE}' not found! 可選: {list(gen2idx.keys())}")

emo_idx = emo2idx[EMOTION]
gen_idx = gen2idx[GENRE]

# 讀取 tokenizer
tokenizer = REMI(params=TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.vocab_size

# 建立模型
if MODEL_TYPE == "transformer":
    model = CondTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=512,
        nlayers=12,
        nhead=16,
        emo_num=len(emo2idx),
        gen_num=len(gen2idx),
        max_seq_len=MAX_LEN,
    ).to(DEVICE)
else:
    model = CondLSTM(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        nlayers=2,
        emo_num=len(emo2idx),
        gen_num=len(gen2idx),
        max_seq_len=MAX_LEN,
    ).to(DEVICE)

# 載入參數
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# 特殊 token id
BOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 0

# 生成
with torch.no_grad():
    seq = [BOS_TOKEN]
    for _ in range(MAX_LEN - 1):
        x = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        emo_t = torch.tensor([emo_idx], dtype=torch.long, device=DEVICE)
        gen_t = torch.tensor([gen_idx], dtype=torch.long, device=DEVICE)
        logits = model(x, emo_t, gen_t)
        logits = logits[0, -1] / TEMPERATURE
        logits[PAD_TOKEN] = float('-inf')
        logits[BOS_TOKEN] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        seq.append(next_token)
        if next_token == EOS_TOKEN:
            break

# 還原成 MIDI
midi = tokenizer(seq)
midi.dump_midi(OUTPUT_PATH)
print(f"Generated: {OUTPUT_PATH}")
