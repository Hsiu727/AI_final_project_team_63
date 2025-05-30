import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import pygame
import pickle

# --- 讀取 labels ---
from generate import get_label_list, generate_music

EMO2IDX_PATH = "emo2idx.pkl"
GEN2IDX_PATH = "gen2idx.pkl"
emotion_list = get_label_list(EMO2IDX_PATH)
genre_list = get_label_list(GEN2IDX_PATH)

class MusicGenGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Music Generator")
        self.geometry("380x300")
        self.resizable(False, False)

        # 標籤選單
        self.lbl1 = tk.Label(self, text="Emotion:")
        self.lbl1.pack(pady=(24,0))
        self.emotion_cb = ttk.Combobox(self, values=emotion_list, state="readonly")
        self.emotion_cb.current(0)
        self.emotion_cb.pack()

        self.lbl2 = tk.Label(self, text="Genre:")
        self.lbl2.pack(pady=(8,0))
        self.genre_cb = ttk.Combobox(self, values=genre_list, state="readonly")
        self.genre_cb.current(0)
        self.genre_cb.pack()

        # 生成按鈕
        self.gen_btn = tk.Button(self, text="Generate", width=16, command=self.async_generate)
        self.gen_btn.pack(pady=20)

        # 狀態訊息
        self.status_label = tk.Label(self, text="", fg="blue")
        self.status_label.pack()

        # 播放相關
        self.play_btn = tk.Button(self, text="Play", command=self.play_music, state="disabled")
        self.play_btn.pack(side="left", padx=30, pady=10)
        self.pause_btn = tk.Button(self, text="Pause", command=self.pause_music, state="disabled")
        self.pause_btn.pack(side="right", padx=30, pady=10)
        self.generated_file = None
        self.music_paused = False

        # 初始化 pygame
        pygame.mixer.init()

    def async_generate(self):
        self.gen_btn.config(state="disabled")
        self.status_label.config(text="Generating...")
        self.play_btn.config(state="disabled")
        self.pause_btn.config(state="disabled")
        t = threading.Thread(target=self.generate_music)
        t.start()

    def generate_music(self):
        emotion = self.emotion_cb.get()
        genre = self.genre_cb.get()
        out_path = "gui_generated.mid"
        try:
            generate_music(emotion, genre, output_path=out_path)
            self.generated_file = out_path
            self.status_label.config(text=f"Generated: {out_path}")
            self.play_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))
            self.status_label.config(text="Generation failed.")
        finally:
            self.gen_btn.config(state="normal")

    def play_music(self):
        if self.generated_file and os.path.exists(self.generated_file):
            try:
                pygame.mixer.music.load(self.generated_file)
                pygame.mixer.music.play()
                self.status_label.config(text=f"Playing: {self.generated_file}")
                self.play_btn.config(state="disabled")
                self.pause_btn.config(state="normal")
            except Exception as e:
                messagebox.showerror("Play Error", str(e))
        else:
            messagebox.showerror("Play Error", "No generated MIDI file.")

    def pause_music(self):
        if pygame.mixer.music.get_busy():
            if not self.music_paused:
                pygame.mixer.music.pause()
                self.music_paused = True
                self.status_label.config(text="Paused.")
                self.pause_btn.config(text="Resume")
            else:
                pygame.mixer.music.unpause()
                self.music_paused = False
                self.status_label.config(text=f"Playing: {self.generated_file}")
                self.pause_btn.config(text="Pause")

if __name__ == "__main__":
    app = MusicGenGUI()
    app.mainloop()
