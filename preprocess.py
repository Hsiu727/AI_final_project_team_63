import os
import re
import pickle
from miditok import REMI  # 你可以選用 CP, Octuple, MuMIDI, 但 REMI最常見
import pretty_midi

# ----------- 1. 檔名解析 -------------
def parse_emo_gen(filename):
    # 支援 XMIDI_happy_country_G05AFQN7.midi
    m = re.match(r'.*_(?P<emo>[A-Za-z]+)_(?P<gen>[A-Za-z]+)_', filename)
    if not m:
        raise ValueError(f"檔名 {filename} 不符 _emo_gen_ 格式")
    return m.group('emo'), m.group('gen')

# ----------- 2. miditok 設定 -------------
# 指定你的 midi 檔資料夾
DATA_DIR = 'mid'
SAVE_PATH = 'dataset.pkl'

# 初始化 tokenizer
tokenizer = REMI()  # 參數可以自定義，有需要可參考 miditok 官網
# 預設 config 足夠小專案測試，進階可再調

# ----------- 3. 開始處理 -------------
dataset = []
emos, gens = set(), set()
for fn in os.listdir(DATA_DIR):
    if not fn.lower().endswith(('.mid', '.midi')):
        continue
    emo, gen = parse_emo_gen(fn)
    midi_path = os.path.join(DATA_DIR, fn)
    try:
        tokens = tokenizer(midi_path)
        if isinstance(tokens, list):
            # 多軌的話 tokens 是 list（每個樂器一條序列）
            all_ids = []
            for t in tokens:
                all_ids.extend(t.ids)
            int_tokens = all_ids
        else:
            int_tokens = tokens.ids
    except Exception as e:
        print(f"[!] 處理 {fn} 失敗：{e}")
        continue
    dataset.append({'tokens': int_tokens, 'emotion': emo, 'genre': gen})
    emos.add(emo)
    gens.add(gen)

# 儲存tokenizer
tokenizer.save_params("tokenizer.json")

# ----------- 4. 儲存處理後資料 -------------
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(dataset, f)

# ----------- 5. 儲存標籤列表 -------------
with open('emo2idx.pkl', 'wb') as f:
    pickle.dump({e: i for i, e in enumerate(sorted(emos))}, f)
with open('gen2idx.pkl', 'wb') as f:
    pickle.dump({g: i for i, g in enumerate(sorted(gens))}, f)

print(f"完成！共處理 {len(dataset)} 首曲子，情緒: {emos}，種類: {gens}")
