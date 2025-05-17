import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # 移除標點
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_csv(input_csv, output_csv, desc_column="description"):
    df = pd.read_csv(input_csv)
    df[desc_column] = df[desc_column].astype(str).apply(clean_text)
    df.to_csv(output_csv, index=False)
    print(f"前處理完成，輸出：{output_csv}")

if __name__ == "__main__":
    preprocess_csv("dataset.csv", "dataset_clean.csv") # need modified | name of dataset.csv 
