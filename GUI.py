import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pygame
import threading

# 初始化 pygame mixer
pygame.mixer.init()

class MusicGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("互動式音樂生成器")
        self.root.geometry("600x300")
        
        # 風格、情緒、類型選項
        self.styles = ["Jazz", "Pop", "Classical", "Rock"]
        self.emotions = ["Happy", "Sad", "Calm", "Energetic"]
        self.types = ["Melody", "Accompaniment", "Full Track"]

        # 建立下拉選單
        ttk.Label(root, text="風格：").grid(column=0, row=0, padx=10, pady=10, sticky="W")
        self.style_cb = ttk.Combobox(root, values=self.styles, state="readonly")
        self.style_cb.grid(column=1, row=0, padx=10, pady=10)
        self.style_cb.current(0)

        ttk.Label(root, text="情緒：").grid(column=0, row=1, padx=10, pady=10, sticky="W")
        self.emotion_cb = ttk.Combobox(root, values=self.emotions, state="readonly")
        self.emotion_cb.grid(column=1, row=1, padx=10, pady=10)
        self.emotion_cb.current(0)

        ttk.Label(root, text="類型：").grid(column=0, row=2, padx=10, pady=10, sticky="W")
        self.type_cb = ttk.Combobox(root, values=self.types, state="readonly")
        self.type_cb.grid(column=1, row=2, padx=10, pady=10)
        self.type_cb.current(0)

        # 生成按鈕
        self.generate_btn = ttk.Button(root, text="生成", command=self.generate_music)
        self.generate_btn.grid(column=0, row=3, columnspan=2, pady=20)

        # 音樂播放器區域
        self.player_frame = ttk.LabelFrame(root, text="播放器")
        self.player_frame.grid(column=2, row=0, rowspan=4, padx=20, pady=10, sticky="NSEW")

        self.play_btn = ttk.Button(self.player_frame, text="播放", command=self.play_music, state="disabled")
        self.play_btn.grid(column=0, row=0, padx=10, pady=10)
        self.pause_btn = ttk.Button(self.player_frame, text="暫停", command=self.pause_music, state="disabled")
        self.pause_btn.grid(column=1, row=0, padx=10, pady=10)

        self.status_label = ttk.Label(self.player_frame, text="尚未生成音樂")
        self.status_label.grid(column=0, row=1, columnspan=2, pady=10)

        # 儲存生成檔案路徑
        self.generated_file = None
        self.is_paused = False

    def generate_music(self):
        style = self.style_cb.get()
        emotion = self.emotion_cb.get()
        mtype = self.type_cb.get()
        # 這裡放入呼叫後端模型生成音樂的程式碼
        # 目前以手動選檔模擬
        path = filedialog.askopenfilename(title="選擇生成的音樂檔案", filetypes=[("MIDI files", "*.mid *.midi"), ("Audio files", "*.mp3 *.wav")])
        if not path:
            return
        self.generated_file = path
        self.status_label.config(text=f"已選擇：{path.split('/')[-1]}")
        self.play_btn.config(state="normal")
        self.pause_btn.config(state="disabled")

    def play_music(self):
        if not self.generated_file:
            return
        try:
            pygame.mixer.music.load(self.generated_file)
            pygame.mixer.music.play()
            self.is_paused = False
            self.play_btn.config(state="disabled")
            self.pause_btn.config(state="normal")
            self.status_label.config(text="播放中")
        except Exception as e:
            messagebox.showerror("播放錯誤", str(e))

    def pause_music(self):
        if not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.pause_btn.config(text="繼續")
            self.status_label.config(text="已暫停")
        else:
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.pause_btn.config(text="暫停")
            self.status_label.config(text="播放中")

if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGeneratorApp(root)
    root.mainloop()
