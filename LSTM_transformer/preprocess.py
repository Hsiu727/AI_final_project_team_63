import os
import pickle
from miditok import REMI, TokenizerConfig

data_dir = "midi_genre_classifier\data"

TOKENIZER_PARAMS = {
    "use_programs": True,
    "one_token_stream_for_programs": True,
    "use_time_signatures": True
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

tokenizer = REMI(config) 

dataset = []
emo2idx, gen2idx = {}, {}

for fname in os.listdir(data_dir):
    if not fname.lower().endswith('.midi'):
        continue
    midi_path = os.path.join(data_dir, fname)
    try:
        # 檔名解析
        parts = fname.split('_')
        if len(parts) < 4:
            print(f"檔名 {fname} 解析失敗")
            continue
        _, emo, genre, _ = parts
        emo2idx.setdefault(emo, len(emo2idx))
        gen2idx.setdefault(genre, len(gen2idx))

        token_objs = tokenizer(midi_path)
        if not isinstance(token_objs, list):
            token_objs = [token_objs]
        tokens_obj = token_objs[0]
        # robust 型態檢查
        if hasattr(tokens_obj, "ids"):
            tokens = tokens_obj.ids
        elif isinstance(tokens_obj, int):
            tokens = [tokens_obj]
        else:
            print(f"{fname}: 未知型態 {type(tokens_obj)}，跳過")
            continue

        dataset.append({
            'tokens': tokens,
            'emotion': emo,
            'genre': genre,
            'filename': fname
        })
    except Exception as e:
        print(f"Error processing {fname}: {e}")


# ###########debug###########
# print(f"\n共收集 {len(dataset)} 首歌")
# print(f"情緒標籤: {emo2idx}")
# print(f"類型標籤: {gen2idx}")
# for i, song in enumerate(dataset[:3]):
#     print(f"\n第{i+1}首：{song['filename']}")
#     print("  tokens 數量 =", len(song['tokens']), "前10 tokens =", song['tokens'][:10])
#     print(f"  標註 emotion = {song['emotion']}, genre = {song['genre']}")

#     print("前30個token對應事件：")
#     for tid in dataset[0]['tokens'][:10]:
#         print(tokenizer[tid])



# # 儲存資料與標籤對應
with open("dataset_fullband.pkl", "wb") as f:
    pickle.dump(dataset, f)
with open("emo2idx.pkl", "wb") as f:
    pickle.dump(emo2idx, f)
with open("gen2idx.pkl", "wb") as f:
    pickle.dump(gen2idx, f)

# # 儲存 tokenizer config
tokenizer.save_params("tokenizer.json")

# for tid in tokens[:2000]:
#     print(tokenizer[tid])


# # 1. 載入 dataset
# with open("dataset_fullband.pkl", "rb") as f:
#     dataset = pickle.load(f)

# # 2. 載入 tokenizer
# tokenizer = REMI(params="tokenizer.json")

# # 3. 選一首 sample（例如第一首）
# tokens = dataset[0]['tokens']
# print(f"tokens length: {len(tokens)}")

# # 4. 解回 MIDI（注意 miditok v3.x 推薦傳入 list of list）
# midi_obj = tokenizer(tokens)  # 這裡 tokens 要用 list of int

# # 5. 輸出為 MIDI 檔案
# midi_obj.dump_midi("test_sample_from_dataset.mid")
# print("已輸出 test_sample_from_dataset.mid，請用 MIDI 軟體聽聽看！")
###########debug###########