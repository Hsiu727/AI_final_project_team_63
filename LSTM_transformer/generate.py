import torch
import pickle
from model import CondTransformer
from miditok import REMI
import sys

def get_label_list(pkl_path):
    with open(pkl_path, "rb") as f:
        label_dict = pickle.load(f)
    sorted_labels = sorted(label_dict, key=lambda k: label_dict[k])
    return sorted_labels

def generate_music(emotion, genre, output_path="generated_output.mid", temperature=1.2, max_len=1024):
    CKPT_PATH = "best_model.pt"
    TOKENIZER_PATH = "tokenizer.json"
    EMO2IDX_PATH = "emo2idx.pkl"
    GEN2IDX_PATH = "gen2idx.pkl"
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    PAD_TOKEN = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    with open(EMO2IDX_PATH, "rb") as f:
        emo2idx = pickle.load(f)
    with open(GEN2IDX_PATH, "rb") as f:
        gen2idx = pickle.load(f)

    if emotion not in emo2idx:
        raise ValueError(f"Emotion '{emotion}' not found in emo2idx! Available: {list(emo2idx.keys())}")
    if genre not in gen2idx:
        raise ValueError(f"Genre '{genre}' not found in gen2idx! Available: {list(gen2idx.keys())}")
    emo_idx = emo2idx[emotion]
    gen_idx = gen2idx[genre]

    tokenizer = REMI(params=TOKENIZER_PATH)
    VOCAB_SIZE = tokenizer.vocab_size

    model = CondTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=512,
        nlayers=12,
        nhead=16,
        emo_num=len(emo2idx),
        gen_num=len(gen2idx),
        max_seq_len=1024,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        seq = [BOS_TOKEN]
        for _ in range(max_len - 1):
            x = torch.tensor([seq], dtype=torch.long, device=DEVICE)
            emo_t = torch.tensor([emo_idx], dtype=torch.long, device=DEVICE)
            gen_t = torch.tensor([gen_idx], dtype=torch.long, device=DEVICE)
            logits = model(x, emo_t, gen_t)
            logits = logits[0, -1] / temperature
            logits[PAD_TOKEN] = float('-inf')
            logits[BOS_TOKEN] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            seq.append(next_token)
            if next_token == EOS_TOKEN:
                break

    midi = tokenizer(seq)
    midi.dump_midi(output_path)
    print(f"Generated: {output_path}")
    return output_path

if __name__ == "__main__":
    # CLI：可用 python generate.py emotion genre [output_path]
    emo_list = get_label_list("emo2idx.pkl")
    gen_list = get_label_list("gen2idx.pkl")
    print("Emotion labels:", emo_list)
    print("Genre labels:", gen_list)
    # 預設值
    default_emotion = emo_list[0]
    default_genre = gen_list[0]
    output_path = "generated_output.mid"
    # 支援命令列參數
    if len(sys.argv) >= 3:
        emotion = sys.argv[1]
        genre = sys.argv[2]
        if len(sys.argv) >= 4:
            output_path = sys.argv[3]
    else:
        print(f"Usage: python generate.py <emotion> <genre> [output_path]")
        print(f"Default: emotion={default_emotion}, genre={default_genre}, output={output_path}")
        emotion = default_emotion
        genre = default_genre
    # 生成
    generate_music(emotion, genre, output_path=output_path)
