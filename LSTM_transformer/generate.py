import argparse
import torch
from model import CondTransformer, CondLSTM
from miditok import REMI
import pickle

# ----- 參數設定與命令列參數解析 -----
parser = argparse.ArgumentParser(description="Generate music MIDI via trained model")
parser.add_argument("--model", choices=["transformer","lstm"], default="transformer",
                    help="Model type to use for generation")
parser.add_argument("--ckpt", type=str, default=None,
                    help="Checkpoint path; default best_{model}.pt")
parser.add_argument("--emotion", type=str, required=True,
                    help="Emotion condition (must be in emo2idx)")
parser.add_argument("--genre", type=str, required=True,
                    help="Genre condition (must be in gen2idx)")
parser.add_argument("--max_len", type=int, default=512,
                    help="Maximum generation length including BOS/EOS")
parser.add_argument("--temperature", type=float, default=1.2,
                    help="Sampling temperature")
parser.add_argument("--output", type=str, default="generated_output.mid",
                    help="Output MIDI file path")
args = parser.parse_args()

# ----- 固定參數 -----
TOKENIZER_PATH = "tokenizer.json"
EMO2IDX_PATH   = "emo2idx.pkl"
GEN2IDX_PATH   = "gen2idx.pkl"
BOS_TOKEN      = 1
EOS_TOKEN      = 2
PAD_TOKEN      = 0
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# 自動決定 checkpoint
ckpt_path = args.ckpt or f"best_{args.model}.pt"

# ----- 載入 tokenizer 與映射表 -----
tokenizer = REMI(params=TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.vocab_size
with open(EMO2IDX_PATH, "rb") as f:
    emo2idx = pickle.load(f)
with open(GEN2IDX_PATH, "rb") as f:
    gen2idx = pickle.load(f)

# ----- 建立並載入模型 -----
if args.model == "transformer":
    model = CondTransformer(
        vocab_size=VOCAB_SIZE,
        emo_num=len(emo2idx),
        gen_num=len(gen2idx),
        d_model=256,
        nlayers=6,
        nhead=8,
        max_seq_len=args.max_len
    )
else:
    model = CondLSTM(
        vocab_size=VOCAB_SIZE,
        emo_num=len(emo2idx),
        gen_num=len(gen2idx),
        event_type_num=None,  # adjust if using event types
        d_model=256,
        hidden_size=512,
        num_layers=2,
        max_seq_len=args.max_len,
        dropout=0.1
    )
model.to(DEVICE)
state = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ----- 驗證輸入條件 -----
emotion = args.emotion
genre   = args.genre
if emotion not in emo2idx:
    raise ValueError(f"Emotion '{emotion}' not found in mapping")
if genre not in gen2idx:
    raise ValueError(f"Genre '{genre}' not found in mapping")
emo_idx = emo2idx[emotion]
gen_idx = gen2idx[genre]

# ----- 開始生成 -----
seq = [BOS_TOKEN]
with torch.no_grad():
    for _ in range(args.max_len - 1):
        x     = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        emo_t = torch.tensor([emo_idx], dtype=torch.long, device=DEVICE)
        gen_t = torch.tensor([gen_idx], dtype=torch.long, device=DEVICE)
        logits = model(x, emo_t, gen_t)
        logits = logits[0, -1] / args.temperature
        # 避免採樣到不可用 token
        logits[BOS_TOKEN] = float('-inf')
        logits[PAD_TOKEN] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        seq.append(next_token)
        if next_token == EOS_TOKEN:
            break

print(f"Generated sequence length: {len(seq)} tokens")

# ----- 顯示事件並儲存成 MIDI -----
for tid in seq:
    if 0 <= tid < VOCAB_SIZE:
        print(tokenizer[tid])
    else:
        print(f"Invalid token ID: {tid}")

midi = tokenizer(seq)
midi.dump_midi(args.output)
print(f"Saved generated MIDI to {args.output}")