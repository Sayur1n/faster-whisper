import argparse
import os
import queue
import sys
import tempfile
import time
from typing import Optional, List

import numpy as np
import sounddevice as sd
import soundfile as sf
import soundcard as sc
import pythoncom
from faster_whisper import WhisperModel

# 等价于在控制台里 set KMP_DUPLICATE_LIB_OK=TRUE
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 等价于 set HF_HOME=D:\hf_cache
os.environ["HF_HOME"] = r"D:\hf_cache"
# ----------------------------- GUI -----------------------------
import tkinter as tk
from tkinter import ttk

class CaptionGUI:
    def __init__(self, always_on_top=True, font_size=16, width=900, height=280):
        self.root = tk.Tk()
        self.root.title("Live Captions")
        self.root.geometry(f"{width}x{height}+100+100")
        self.root.attributes("-topmost", always_on_top)
        self.root.configure(bg="#111111")

        # 样式
        self.font_main = ("Consolas", font_size)
        self.font_tail = ("Consolas", max(12, font_size - 2))

        # 主区：已定稿文本（可滚动）
        self.text = tk.Text(self.root, wrap="word", bg="#111111", fg="#eaeaea",
                            insertbackground="#eaeaea", relief="flat", font=self.font_main)
        self.text.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.text.tag_configure("tail", foreground="#9aa0a6", font=self.font_tail)
        self.text.tag_configure("main", foreground="#eaeaea", font=self.font_main)

        # 底部控制
        bar = tk.Frame(self.root, bg="#111111")
        bar.pack(fill="x", padx=8, pady=(0, 8))

        self.var_topmost = tk.BooleanVar(value=always_on_top)
        chk = ttk.Checkbutton(bar, text="置顶", variable=self.var_topmost, command=self.toggle_topmost)
        chk.pack(side="left")

        btn_clear = ttk.Button(bar, text="清空", command=self.clear_all)
        btn_clear.pack(side="left", padx=(8, 0))

        # 键盘快捷键
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<Control-l>", lambda e: self.clear_all())

        # 文本缓存
        self.committed_text = ""
        self.tail_text = ""

        # 线程通信
        self.queue = queue.Queue()
        self.root.after(50, self._poll_queue)

    def toggle_topmost(self):
        self.root.attributes("-topmost", self.var_topmost.get())

    def clear_all(self):
        self.text.delete("1.0", "end")
        self.committed_text = ""
        self.tail_text = ""

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "update":
                    committed, tail = payload
                    self._render(committed, tail)
                elif kind == "append":
                    inc = payload
                    if inc:
                        self.text.insert("end", inc, ("main",))
                        self.text.see("end")
                elif kind == "clear_tail":
                    self._render(self.committed_text, "")
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def _render(self, committed: str, tail: str):
        # 仅当变化时更新，避免闪烁
        if committed != self.committed_text or tail != self.tail_text:
            self.text.delete("1.0", "end")
            if committed:
                self.text.insert("end", committed, ("main",))
            if tail:
                self.text.insert("end", "\n" + tail if committed else tail, ("tail",))
            self.text.see("end")
            self.committed_text = committed
            self.tail_text = tail

    def update_text(self, committed: str, tail: str):
        self.queue.put(("update", (committed, tail)))

    def append_committed(self, inc: str):
        self.queue.put(("append", inc))

    def clear_tail(self):
        self.queue.put(("clear_tail", None))

    def run(self):
        self.root.mainloop()

# ----------------------------- audio utils -----------------------------

def get_samplerate_for_device(device_idx: int, fallback: int = 16000) -> int:
    try:
        info = sd.query_devices(device_idx)
        sr = int(info.get("default_samplerate") or fallback)
        return sr
    except Exception:
        return fallback


def find_default_input_device_index() -> Optional[int]:
    try:
        default_in, _ = sd.default.device
        if isinstance(default_in, int):
            return default_in
    except Exception:
        pass
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            return idx
    return None

# ----------------------------- text helpers -----------------------------

def sanitize_text(text: str) -> str:
    return text.strip()

# === (4-1) 新增：基础规范化/重复折叠/重复度检测 ===
import unicodedata
import re
from collections import Counter  # 目前未直接使用，可保留

def normalize_text_basic(text: str) -> str:
    """统一全/半角、去多余空白、统一标点空格、压缩连串标点。"""
    if not text:
        return text
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([。．.!！?？,，…])", r"\1", t)
    t = re.sub(r"([（(【\[『「])\s+", r"\1", t)
    t = re.sub(r"\s+([）)】\]』」])", r"\1", t)
    t = re.sub(r"[!！]{3,}", "！！", t)
    t = re.sub(r"[?？]{3,}", "？？", t)
    t = re.sub(r"([!！?？])\1{3,}", r"\1\1", t)
    return t.strip()

def collapse_repetitions(text: str) -> str:
    """
    通用重复折叠：
    1) 常见口癖短语连续重复折叠；
    2) 任意2~20字子串的连续重复折叠；
    3) 尾部词干重复折叠（…なさい!!なさい!! → …なさい!!）。
    """
    if not text:
        return text

    t = text
    common_phrases = [
        r"おやすみなさい", r"こんにちは", r"こんばんは", r"どうも(ですね|です)?",
        r"ありがとう(ございます|した)?", r"以上(です|となります)?",
        r"えーと", r"あのー", r"えっと", r"そのー",
        r"你好(啊)?", r"大家好",
        r"hi|hello|hey|good (morning|afternoon|evening)",
        r"thanks( a lot)?|thank you( very much)?|that'?s (all|it)"
    ]
    for p in common_phrases:
        t = re.sub(rf"((?:{p})(?:[!！?？。．])*)(\s*\1)+", r"\1", t, flags=re.IGNORECASE)

    t = re.sub(r"(.{2,20}?)(\1){1,}", r"\1", t)

    t = re.sub(r"(なさい)([!！])?\s*(\1([!！])?\s*){1,}$", r"\1\2", t)

    t = re.sub(r"[!！]{3,}", "！！", t)
    t = re.sub(r"[?？]{3,}", "？？", t)

    return t.strip()

def too_repetitive(text: str, threshold: float = 0.65) -> bool:
    """
    简易2-gram多样性判定：多样性过低→视为重复噪声。
    threshold 越大越严格（0.65 是一个偏稳的值）。
    """
    s = text.replace(" ", "")
    if len(s) < 12:
        return False
    grams = [s[i:i+2] for i in range(len(s)-1)]
    if not grams:
        return False
    uniq = len(set(grams))
    div = uniq / len(grams)
    return div < (1 - threshold)

# === (1) 强化问候/套话过滤（按句清理） ===
_GREETING_PATTERNS = [
    # --- Japanese common greetings / closings / fillers ---
    r"^(みなさん|皆さん)?(こんにちは|こんばんは|おはよう(ございます)?)[！!。\.]?$",
    r"^(どうも|どうもです|どうもですね|ありがとうございます|ありがとうございました)[！!。\.]?$",
    r"^(ご清聴ありがとうございました|ご視聴ありがとうございました|以上(です|となります))[！!。\.]?$",
    r"^(えーと|あのー|えっと|そのー)[、, ]*$",
    # --- Chinese ---
    r"^(大家好|你好(啊)?|各位(朋友|同学|老师)好)[！!。\.]?$",
    r"^(谢谢(大家)?|多谢|感谢观看|以上(就是)?(全部)?(内容)?)$",
    # --- English ---
    r"^(hi|hello|hey|good (morning|afternoon|evening))[\!\.\s]*$",
    r"^(thanks( a lot)?|thank you( very much)?|that'?s (all|it))[\!\.\s]*$",
]
_GREETING_RE = [re.compile(p, re.IGNORECASE) for p in _GREETING_PATTERNS]

def strip_greetings_sentences(text: str) -> str:
    """
    对文本按句切分，丢弃问候/结束/口癖句子，保留有效内容。
    """
    if not text:
        return text
    splits = re.split(r"([。．.!！?？])", text)
    sents = []
    for i in range(0, len(splits), 2):
        seg = splits[i].strip()
        punc = splits[i+1] if i+1 < len(splits) else ""
        if not seg:
            continue
        sent = (seg + punc).strip()
        if any(rgx.match(seg) or rgx.match(sent) for rgx in _GREETING_RE):
            continue
        sents.append(sent)
    joined = "".join(sents).strip()
    return joined

# 保留旧函数（不再调用），防止你其他地方依赖
def apply_blacklist(text: str) -> str:
    t = sanitize_text(text)
    blacklist = [
        "ご視聴ありがとうございました",
        "ご清聴ありがとうございました",
        "ありがとうございました",
        "以上です",
        "以上となります",
        "視聴ありがとうございました",
    ]
    for p in blacklist:
        if t == p or (len(t) <= len(p) + 2 and p in t):
            return ""
        if any([a == p or (len(a) <= len(p) + 2 and p in a) for a in t.split('、')]):
            return ""
        if any([a == p or (len(a) <= len(p) + 2 and p in a) for a in t.split('。')]):
            return ""
    return text

def diff_append(old: str, new: str) -> str:
    if not old:
        return new
    max_overlap = min(len(old), len(new))
    for k in range(max_overlap, -1, -1):
        if old.endswith(new[:k]):
            return new[k:]
    return new

# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live captions with GUI (system loopback / mic) + rollback stabilization"
    )
    parser.add_argument("--source", choices=["mic", "system"], default="system",
                        help="音频来源：mic=麦克风（sounddevice）；system=系统回环（soundcard）")
    parser.add_argument("--model", default="distil-large-v3",
                        help="Whisper/faster-whisper 模型名，如 distil-large-v3/large-v3/large-v2 等")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute_type", default="float16",
                        help='cuda用 "float16"；CPU 可用 "int8" / "int8_float16"')
    parser.add_argument("--beam-size", type=int, default=5)  # （2）更稳的beam
    parser.add_argument("--lang", default=None)
    parser.add_argument("--chunk-sec", type=float, default=0.6, help="采集块大小（秒）")
    parser.add_argument("--accum-sec", type=float, default=2.0, help="转写窗口长度（秒）")
    parser.add_argument("--overlap-sec", type=float, default=0.5, help="窗口重叠（秒）")
    parser.add_argument("--stabilize-tail-sec", type=float, default=0.7, help="尾部可回滚区（秒）")
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--min-silence-ms", type=int, default=800)
    parser.add_argument("--vol-gain", type=float, default=1.0)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--topmost", action="store_true", help="窗口置顶（默认置顶）")
    parser.add_argument("--no-topmost", dest="topmost", action="store_false")
    parser.set_defaults(topmost=True)
    args = parser.parse_args()

    # GUI
    gui = CaptionGUI(always_on_top=args.topmost, font_size=args.font_size)

    # ASR model
    print(f"[ASR] loading model: {args.model} on {args.device} ({args.compute_type})")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    # queues
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    # audio capture
    stream_ctx = None
    if args.source == "system":
        speaker = sc.default_speaker()
        if speaker is None:
            print("找不到默认扬声器，请检查 Windows 声音设置。", file=sys.stderr)
            sys.exit(1)
        loop_mic = sc.get_microphone(speaker.name, include_loopback=True)

        samplerate = 48000  # 48k 降低断流；Whisper内部会重采样
        channels = 2
        frames_per_chunk = int(samplerate * args.chunk_sec)
        print(f"[Audio] SYSTEM (soundcard): {speaker.name}, sr={samplerate}, ch={channels}, chunk={args.chunk_sec}s")

        import threading
        def producer():
            pythoncom.CoInitialize()
            try:
                with loop_mic.recorder(
                    samplerate=samplerate,
                    channels=channels,
                    blocksize=frames_per_chunk,
                    # exclusive_mode=True,   # 如仍断流可尝试
                ) as rec:
                    while True:
                        data = rec.record(numframes=frames_per_chunk)  # (frames, ch)
                        audio_q.put(data.copy())
            finally:
                pythoncom.CoUninitialize()
        threading.Thread(target=producer, daemon=True).start()

    else:
        device_idx = find_default_input_device_index()
        if device_idx is None:
            print("未找到麦克风输入设备。", file=sys.stderr)
            sys.exit(1)
        samplerate = get_samplerate_for_device(device_idx, fallback=48000)  # 通常 48k
        channels = 1
        blocksize = max(1, int(samplerate * args.chunk_sec))
        print(f"[Audio] MIC (sounddevice): idx={device_idx} ({sd.query_devices(device_idx)['name']}), sr={samplerate}, ch={channels}, chunk={args.chunk_sec}s")

        def callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            audio_q.put(indata.copy())

        try:
            stream_ctx = sd.InputStream(
                device=device_idx,
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
                blocksize=blocksize,
                callback=callback,
            )
        except Exception as e:
            print(f"打开麦克风失败：{e}", file=sys.stderr)
            sys.exit(1)

    # processing params
    window_frames = int(samplerate * args.accum_sec)
    overlap_frames = int(samplerate * args.overlap_sec)
    tail_cut = max(0.0, min(args.stabilize_tail_sec, args.accum_sec))

    buffer: List[np.ndarray] = []
    buffered_frames = 0

    committed_text = ""
    last_tail = ""

    # run processing in a background thread so GUI stays responsive
    import threading

    def processor():
        nonlocal buffer, buffered_frames, committed_text, last_tail
        entered = False
        try:
            if stream_ctx is not None:
                stream_ctx.__enter__()
                entered = True

            while True:
                block = audio_q.get()
                # 主线程处理：转单声道 + 增益
                if block.ndim == 2:
                    mono = block.mean(axis=1)
                else:
                    mono = block.squeeze(-1)
                mono = (mono * args.vol_gain).astype(np.float32, copy=False)

                buffer.append(mono)
                buffered_frames += mono.shape[0]

                if buffered_frames >= window_frames:
                    window = np.concatenate(buffer, axis=0)[:window_frames]
                    remain = window[-overlap_frames:] if overlap_frames > 0 else np.zeros((0,), dtype=np.float32)
                    buffer = [remain]
                    buffered_frames = remain.shape[0]

                    # 写临时 wav
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        wav_path = tmp.name
                    try:
                        sf.write(wav_path, window, samplerate)

                        # （2）上下文策略：短prefix，禁用跨块隐藏状态
                        prefix = committed_text[-24:] if committed_text else ""

                        kw = dict(
                            beam_size=max(3, args.beam_size),
                            temperature=0.0,
                            condition_on_previous_text=False,   # 关键
                            prefix=prefix,                       # 关键
                            word_timestamps=True,
                            # （4）更严的过滤，抑制静音幻听与重复
                            no_speech_threshold=0.75,
                            log_prob_threshold=-0.35,
                            compression_ratio_threshold=2.2,
                        )
                        if args.lang:
                            kw["language"] = args.lang
                        if args.vad:
                            kw["vad_filter"] = True
                            kw["vad_parameters"] = dict(min_silence_duration_ms=args.min_silence_ms)

                        segments, info = model.transcribe(wav_path, **kw)

                        # 聚合文本与词级时间
                        words = []
                        seg_text_all = []
                        for seg in segments:
                            seg_text_all.append(seg.text)
                            if getattr(seg, "words", None):
                                words.extend(seg.words)

                        chunk_text = sanitize_text("".join(seg_text_all))

                        # （4-2）规范化→折叠重复→问候过滤→重复度拦截
                        chunk_text = normalize_text_basic(chunk_text)
                        chunk_text = collapse_repetitions(chunk_text)
                        chunk_text = strip_greetings_sentences(chunk_text)
                        if not chunk_text.strip() or too_repetitive(chunk_text, threshold=0.65):
                            gui.update_text(committed_text, "")
                            continue

                        # 划分“定稿/尾部”
                        chunk_dur = window.shape[0] / samplerate
                        stable_end = max(0.0, chunk_dur - tail_cut)
                        if words:
                            stable_words = [w.word for w in words if (w.end or 0.0) <= stable_end]
                            tail_words = [w.word for w in words if (w.end or 0.0) > stable_end]
                            stable_text = sanitize_text("".join(stable_words))
                            tail_text = sanitize_text("".join(tail_words))
                        else:
                            cut_idx = int(len(chunk_text) * (stable_end / max(chunk_dur, 1e-6)))
                            stable_text = sanitize_text(chunk_text[:cut_idx])
                            tail_text = sanitize_text(chunk_text[cut_idx:])

                        # 打增量（定稿部分）
                        inc = diff_append(committed_text, committed_text + stable_text)
                        if inc:
                            committed_text += stable_text
                            gui.update_text(committed_text, tail_text)
                        else:
                            # 没有新定稿，可能尾部变化
                            if tail_text != last_tail:
                                gui.update_text(committed_text, tail_text)

                        last_tail = tail_text

                    finally:
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass

        except Exception as e:
            print("处理线程异常：", e, file=sys.stderr)
        finally:
            if entered:
                stream_ctx.__exit__(None, None, None)

    threading.Thread(target=processor, daemon=True).start()

    # run GUI loop
    gui.run()


if __name__ == "__main__":
    main()
