import os, threading, time, re, sys, traceback, queue
import numpy as np
import sounddevice as sd
import onnxruntime as ort 
from tkinter import Tk, Frame, Entry, Label, Text, messagebox
from Live2dTK import Live2dFrame
from llama_cpp import Llama
import pykakasi 

# -------------------------------
# 1. 系統偵測與路徑設定
# -------------------------------
def auto_setup_threads():
    cores = os.cpu_count() or 4
    return min(cores - 2, 6) if cores > 4 else max(1, cores - 2)

ASSIGNED_THREADS = auto_setup_threads()

def get_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    p = os.path.normpath(os.path.join(base_path, relative_path))
    if not os.path.exists(p):
        p = os.path.normpath(os.path.join(base_path, "_internal", relative_path))
    return p

# -------------------------------
# 2. 語音引擎 (純 ONNX Runtime 版)
# -------------------------------
class LocalTTS:
    def __init__(self, live2d_frame):
        self.live2d_frame = live2d_frame
        tts_dir = get_path("models/tts1")
        
        # 載入 ONNX 模型 (Tamamo)
        model_path = os.path.join(tts_dir, "tamamo_cross_final.onnx")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # 載入 tokens.txt
        self.symbol_to_id = {}
        with open(os.path.join(tts_dir, "tokens.txt"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) >= 2:
                    self.symbol_to_id[parts[0]] = int(parts[1])
        
        self.kks = pykakasi.kakasi()
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        threading.Thread(target=self._speech_worker, daemon=True).start()

    def enqueue_speech(self, text):
        if text.strip(): self.speech_queue.put(text)

    def _speech_worker(self):
        while True:
            text = self.speech_queue.get()
            if text: self._generate_and_stream(text)
            self.speech_queue.task_done()

    def _text_to_token_ids(self, text):
        """將文字轉為 Token ID 序列 (含羅馬字轉換)"""
        result = self.kks.convert(text)
        romaji = " ".join([item['hepburn'] for item in result])
        
        token_ids = [0, 0] # 前置空格
        for char in romaji:
            if char == ' ': token_ids.extend([0, 0])
            else:
                tid = self.symbol_to_id.get(char, 0)
                token_ids.append(tid)
                token_ids.append(0)
        token_ids.extend([0, 0])
        return np.array([token_ids], dtype=np.int64)

    def _generate_and_stream(self, text):
        if len(text) < 1: return
        
        try:
            # 推理參數
            token_ids = self._text_to_token_ids(text)
            input_lengths = np.array([token_ids.shape[1]], dtype=np.int64)
            sid = np.array([113], dtype=np.int64) 
            
            inputs = {
                "input": token_ids,
                "input_lengths": input_lengths,
                "sid": sid,
                "noise_scale": np.array([0.6], dtype=np.float32),
                "noise_scale_w": np.array([0.667], dtype=np.float32),
                "length_scale": np.array([1.1], dtype=np.float32)
            }
            
            # 運行推理
            raw_audio = self.session.run(None, inputs)[0].squeeze()
            samples = raw_audio.astype(np.float32)
            
            # 🔥 智能切割 (日文專用參數)
            threshold = 0.0005 
            last_point = len(samples)
            for i in range(len(samples)-1, 0, -1):
                if abs(samples[i]) > threshold:
                    last_point = i
                    break
            samples = samples[:last_point+3000] 

            samples *= 1.5 
            rate = 22050 

            # 淡入淡出與播放 (同前)
            fade_len = int(rate * 0.02)
            if len(samples) > fade_len * 2:
                samples[:fade_len] *= np.linspace(0, 1, fade_len)
                samples[-fade_len:] *= np.linspace(1, 0, fade_len)
            
            self.is_speaking = True
            def mouth_loop():
                st = time.time()
                while self.is_speaking:
                    idx = int((time.time() - st) * rate)
                    if idx < len(samples):
                        if (time.time() - st) > 0.1:
                            chunk = samples[idx : idx + 800]
                            v = np.mean(np.abs(chunk)) if len(chunk) > 0 else 0
                            if self.live2d_frame.model:
                                self.live2d_frame.model.SetParameterValue("ParamMouthOpenY", min(v * 40, 1.2), 1.0)
                    else: break
                    time.sleep(0.01)
                if self.live2d_frame.model: self.live2d_frame.model.SetParameterValue("ParamMouthOpenY", 0, 1.0)

            threading.Thread(target=mouth_loop, daemon=True).start()
            sd.play(samples, rate)
            sd.wait()
            self.is_speaking = False
        except: pass

# -------------------------------
# 3. 主 UI 邏輯
# -------------------------------
try:
    root = Tk()
    root.title("ローカルAIガールフレンド")
    root.geometry("1100x800")
    root.configure(bg="#1e1e1e")

    l2d_container = Frame(root, bg="#1e1e1e", width=600, height=800)
    l2d_container.pack(side="right", fill="both", expand=True)

    chat_frame = Frame(root, bg="#252525", width=500)
    chat_frame.pack(side="left", fill="both", padx=15, pady=15)

    chat_text = Text(chat_frame, fg="#FFB6C1", bg="#1a1a1a", font=("Microsoft JhengHei", 12), wrap='word', state="disabled", bd=0)
    chat_text.pack(side="top", fill="both", expand=True, padx=5, pady=5)

    user_input = Entry(chat_frame, font=("Microsoft JhengHei", 15), bg="#111111", fg="white", insertbackground="white", relief="flat", highlightthickness=2, highlightbackground="#444444", highlightcolor="#FFB6C1")
    user_input.pack(side="bottom", fill="x", pady=(30, 20), ipady=20, padx=10) 

    llm = Llama(model_path=get_path("model.gguf"), n_ctx=2048, n_threads=ASSIGNED_THREADS, verbose=False)
    live2d_frame = Live2dFrame(l2d_container, model_path=get_path("live2d/Hiyori/Hiyori.model3.json"), width=600, height=750)
    live2d_frame.pack(expand=True)
    lixue_tts = LocalTTS(live2d_frame)
    
    # 移除翻譯器，全靠 Prompt

    def update_chat(txt):
        chat_text.config(state="normal")
        chat_text.delete("1.0", "end")
        chat_text.insert("end", txt)
        chat_text.see("end")
        chat_text.config(state="disabled")

    def on_send(event=None):
        input_text = user_input.get().strip()
        if not input_text: return
        user_input.delete(0, 'end')
        update_chat("...🐾")
        
        def run_ai():
            try:
                # 🔥 強化 Prompt：確保本地 Qwen 懂中文並只輸出日文
                prompt = (
                    f"<|im_start|>system\n"
                    f"你要扮演使用者的女友。無論使用者說什麼語言，你都必須僅使用「日文」俏皮可愛地回答。\n"
                    f"🐾<|im_end|>\n"
                    f"<|im_start|>user\n{input_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                stream = llm(prompt, max_tokens=150, stop=["<|im_end|>"], stream=True)
                
                full_txt, buffer = "", ""
                soft_punc = ['，', '、', ',']
                hard_punc = ['。', '！', '？', '；', '\n', '.', '!', '?'] 

                for chunk in stream:
                    token = chunk["choices"][0].get("text", "")
                    full_txt += token
                    buffer += token
                    root.after(0, lambda t=full_txt: update_chat(t))
                    
                    if any(p in token for p in hard_punc) or (len(buffer) > 15 and any(p in token for p in soft_punc)):
                        to_say = buffer.strip()
                        # 🔥 本地安全網：如果包含日文字元才送去語音，防止空包彈
                        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', to_say):
                            lixue_tts.enqueue_speech(to_say)
                            buffer = ""
                
                # 最後處理
                if buffer.strip() and re.search(r'[\u3040-\u309F\u30A0-\u30FF]', buffer):
                    lixue_tts.enqueue_speech(buffer.strip())
            except: pass

        threading.Thread(target=run_ai, daemon=True).start()

    user_input.bind("<Return>", on_send)
    user_input.focus_set()
    root.mainloop()
except Exception:
    messagebox.showerror("啟動失敗", traceback.format_exc())
