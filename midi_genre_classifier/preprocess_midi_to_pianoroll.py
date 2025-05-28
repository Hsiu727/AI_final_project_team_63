import os
import numpy as np
from tqdm import tqdm
from midi_dataset import midi_to_pianoroll, extract_emotion_genre_from_filename

# 設定參數
DATA_ROOT = 'data'
OUT_DIR = 'preprocessed'
MAX_LENGTH = 500  # 和訓練對齊

os.makedirs(OUT_DIR, exist_ok=True)
meta = []

for fname in tqdm(os.listdir(DATA_ROOT)):
    if not fname.endswith('.midi'): continue
    fpath = os.path.join(DATA_ROOT, fname)
    # 自訂的 midi to pianoroll 轉換
    try:
        pr = midi_to_pianoroll(fpath, max_length=MAX_LENGTH)  # [128, MAX_LENGTH]
        if pr.shape[1] < MAX_LENGTH:
            # 補 0（右補）
            pr = np.pad(pr, ((0,0),(0,MAX_LENGTH-pr.shape[1])), 'constant')
        elif pr.shape[1] > MAX_LENGTH:
            pr = pr[:,:MAX_LENGTH]
    except Exception as e:
        print(f"Fail: {fname}, err={e}")
        continue
    # 抓 label
    emotion, genre = extract_emotion_genre_from_filename(fname)
    # 存 numpy
    np.save(os.path.join(OUT_DIR, fname + '.npy'), pr.astype(np.float32))
    meta.append([fname, emotion, genre])

# 儲存 meta 標註檔
import pandas as pd
meta_df = pd.DataFrame(meta, columns=['fname','emotion','genre'])
meta_df.to_csv(os.path.join(OUT_DIR, 'meta.csv'), index=False)
print("預處理完成！")
