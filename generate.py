# generate.py
import torch
import pickle
from model import CondTransformer
from miditok import REMI
from pathlib import Path


# ------------------------
# 設定路徑與參數
# ------------------------
CKPT_PATH      = 'checkpoints/epoch10.pt' # 根據epoch數更改
EMO_PKL        = 'emo2idx.pkl'
GEN_PKL        = 'gen2idx.pkl'
TOKENIZER_JSON = "tokenizer.json"
OUTPUT_MIDI    = 'generated.mid'

SOS_TOKEN = 1  # <SOS> token id，請根據 tokenizer/vocab 設定
EOS_TOKEN = 2  # <EOS> token id，請根據 tokenizer/vocab 設定
MAX_LEN   = 512 # 要跟train.py一樣
TOP_K     = 4

# ------------------------
# 載入標籤對應 & tokenizer
# ------------------------
with open(EMO_PKL, 'rb') as f:
    emo2idx = pickle.load(f)
with open(GEN_PKL, 'rb') as f:
    gen2idx = pickle.load(f)

tokenizer = REMI(params=TOKENIZER_JSON) if TOKENIZER_JSON else REMI()
VOCAB_SIZE = tokenizer.vocab_size

# ------------------------
# 載入模型
# ------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CondTransformer(
    vocab_size=VOCAB_SIZE,
    emo_num=len(emo2idx),
    gen_num=len(gen2idx),
    d_model=256,
    nhead=8,
    nlayers=4,
    max_seq_len=MAX_LEN
).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

# ------------------------
# 條件生成函數
# ------------------------
@torch.no_grad()
def generate(model, tokenizer, emotion_idx, genre_idx, max_len=256, sos_token=1, eos_token=2, device='cpu', top_k=4):
    tokens = [sos_token]
    for _ in range(max_len):
        input_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
        emo = torch.tensor([emotion_idx], dtype=torch.long, device=device)
        gen = torch.tensor([genre_idx], dtype=torch.long, device=device)
        logits = model(input_tokens, emo, gen)  # [1, L, vocab_size]
        next_token_logits = logits[0, -1]       # [vocab_size]
        # Top-k sampling
        topk_vals, topk_indices = torch.topk(next_token_logits, k=min(top_k, VOCAB_SIZE))
        probs = torch.softmax(topk_vals, dim=-1)
        next_token = topk_indices[torch.multinomial(probs, 1).item()].item()
        tokens.append(next_token)
        if next_token == eos_token:
            break
    return tokens

# ------------------------
# tokens 轉 MIDI
# ------------------------
def tokens_to_midi(tokenizer, tokens, output_path):
    # 建立 TokenSequence 給 miditok
    tokens = [t for t in tokens if t != 0]
    if tokens and tokens[0] == 1:  # BOS
        tokens = tokens[1:]
    midi = tokenizer([tokens])  # miditok v3.x：tokenizer() 會自動處理 list/int list
    midi.dump_midi(Path(output_path))

# ------------------------
# 主程式：輸入條件與生成
# ------------------------
if __name__ == '__main__':
    # 你可以改成命令列參數，這裡直接寫死
    user_emo = 'happy'
    user_gen = 'pop'
    output_path = OUTPUT_MIDI

    # 找對應 index
    emotion_idx = emo2idx[user_emo]
    genre_idx = gen2idx[user_gen]

    gen_tokens = generate(
        model, tokenizer, emotion_idx, genre_idx,
        max_len=MAX_LEN, sos_token=SOS_TOKEN, eos_token=EOS_TOKEN,
        device=device, top_k=TOP_K
    )

    print('生成完成，token 長度：', len(gen_tokens))


    #debugging
    print("tokenizer.vocab:", tokenizer.vocab)

    print(tokenizer.special_tokens)

    print("Generated tokens:", gen_tokens)
    print("Max token id:", max(gen_tokens))
    print("Min token id:", min(gen_tokens))
    print("Vocab size:", tokenizer.vocab_size)

    tokens_to_midi(tokenizer, gen_tokens, output_path)
    print('MIDI 已儲存：', output_path)
