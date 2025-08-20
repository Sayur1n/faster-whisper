# -*- coding: utf-8 -*-
"""
Real-time ASR (mic/system) + pause-based punctuation + incremental translation (Youdao createRequest) + GUI
- 终止就清空（防重复）
- 终止条件：句号 / ≥5个逗号 / 5秒无新segments
- GUI 两个文本框展示，未确定（tentative）为淡灰色大字号，确定（committed）为黑色
"""

import argparse
import os
import queue
import sys
import tempfile
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import soundcard as sc
from soundcard import mediafoundation as mf  # 确保线程内可初始化/释放 COM（Windows）
from faster_whisper import WhisperModel
import pythoncom
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkfont
from datetime import datetime

# 直接使用你提供的有道 Demo（含 APP_KEY/APP_SECRET）
try:
    from apidemo.TranslateDemo import createRequest
except Exception as e:
    print("导入 apidemo.TranslateDemo.createRequest 失败，请确认文件路径与模块可见性。", file=sys.stderr)
    raise

# 推荐：减少 MKL 与 HuggingFace 缓存冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", r"D:\hf_cache")


# ---------------- Utils: devices & text ----------------
def sanitize_text(text: str) -> str:
    return (text or "").strip()


def apply_blacklist(text: str) -> str:
    t = sanitize_text(text)
    t_lower = t.lower()
    blacklist = [
        "ご視聴ありがとうございました、",
        "ご視聴ありがとうございました",
        "ご清聴ありがとうございました",
        "ありがとうございました",
        "以上です",
        "以上となります",
        "視聴ありがとうございました",
        "次回予告",
        "次回もお楽しみに",
        "チャンネル登録お願いします",
        "高評価よろしくお願いします",
        "それでは、また会いましょう",
        "この動画の字幕は視聴者によって作成されました",
        "お待ちしております",
        # 常见幻听尾句/署名（大小写不敏感）
        "thanks for watching",
        "thanksforwatching",
        "teksting av nicolai winther",
        "subtitle by",
        "subtitles by",
        "captioned by"
    ]
    for p in blacklist:
        p_lower = p.lower()
        # 全文近似匹配 或 句读内的短句匹配
        if t_lower == p_lower or (len(t_lower) <= len(p_lower) + 2 and p_lower in t_lower):
            return ""
        if any([(a.lower() == p_lower or (len(a) <= len(p) + 2 and p_lower in a.lower())) for a in t.split('、')]):
            return ""
        if any([(a.lower() == p_lower or (len(a) <= len(p) + 2 and p_lower in a.lower())) for a in t.split('。')]):
            return ""
    return text


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


# ---------------- Pause-based punctuation ----------------
def _ends_with_any(s: str, chars: str) -> bool:
    s = (s or "").rstrip()
    return bool(s) and s[-1] in set(chars)

def _starts_with_any(s: str, chars: str) -> bool:
    s = (s or "").lstrip()
    return bool(s) and s[0] in set(chars)

def punctuate_segments(
    segments,
    lang_hint: str = None,
    short_pause: float = 0.35,
    long_pause: float = 0.95,
) -> str:
    """
    将 segments 按词级时间拼接；基于相邻词/段的时间差添加顿号/句号。
    """
    lang = (lang_hint or "").lower()
    is_cjk = any(x in lang for x in ["zh", "ja", "ko"])
    COMMA = "、" if is_cjk else ","
    PERIOD = "。" if is_cjk else "."

    out = []
    prev_end = None
    TAIL = f"{COMMA}{PERIOD}，,。.!?！？、"
    for seg in segments:
        words = getattr(seg, "words", None)
        if words:
            for w in words:
                token = (w.word or "").strip()
                if not token:
                    continue
                if prev_end is not None and w.start is not None:
                    gap = w.start - prev_end

                    if gap >= long_pause:
                        if out and not _ends_with_any("".join(out), TAIL):
                            out.append(PERIOD)
                    elif gap >= short_pause:
                        if out and not _ends_with_any("".join(out), TAIL):
                            out.append(COMMA)
                out.append(token)
                prev_end = w.end or w.start
        else:
            seg_text = (seg.text or "").strip()
            if seg_text:
                if prev_end is not None and seg.start is not None:
                    gap = seg.start - prev_end
                    if gap >= long_pause:
                        if out and not _ends_with_any("".join(out), TAIL):
                            out.append(PERIOD)
                    elif gap >= short_pause:
                        if out and not _ends_with_any("".join(out), TAIL):
                            out.append(COMMA)
                out.append(seg_text)
                prev_end = seg.end or prev_end

    # 末尾兜底：若没有终止符，补一个顿号，利于翻译小停顿
    s = "".join(out)
    if s and not _ends_with_any(s, TAIL):
        out.append(COMMA)   # CJK 用 '、'，英文字母路径 COMMA 会是 ','
    return "".join(out).strip()



# ---------------- GUI ----------------
class CaptionGUI:
    """
    上下布局：
    - 上：原文 (ASR)
    - 下：译文 (Youdao)
    窗口 600x400，上下等高，文字过多时自动下翻
    """
    def __init__(self, title="ASR + Translation (Youdao)"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("600x400")

        # 字体
        self.font_main = tkfont.Font(family="Segoe UI", size=16)

        # 整体布局：3 行
        # row 0 = 顶部状态栏
        # row 1 = 原文框
        # row 2 = 译文框
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.columnconfigure(0, weight=1)

        # 顶部状态栏
        top = tk.Frame(self.root)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        self.lang_label = tk.Label(top, text="[lang: ?  p=?]", anchor="w", font=("Segoe UI", 11))
        self.lang_label.pack(side=tk.LEFT)
        self.status_label = tk.Label(top, text="准备就绪", anchor="e", font=("Segoe UI", 11))
        self.status_label.pack(side=tk.RIGHT)

        # 原文 frame
        frame_src = tk.Frame(self.root)
        frame_src.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        frame_src.rowconfigure(1, weight=1)
        frame_src.columnconfigure(0, weight=1)

        tk.Label(frame_src, text="原文 (ASR)", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 2)
        )
        self.text_src = ScrolledText(frame_src, wrap="word")
        self.text_src.grid(row=1, column=0, sticky="nsew")
        self.text_src.configure(font=self.font_main)
        self.text_src.tag_configure("tentative", foreground="#999999")
        self.text_src.tag_configure("committed", foreground="#000000")

        # 译文 frame
        frame_dst = tk.Frame(self.root)
        frame_dst.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 6))
        frame_dst.rowconfigure(1, weight=1)
        frame_dst.columnconfigure(0, weight=1)

        tk.Label(frame_dst, text="译文 (Youdao)", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 2)
        )
        self.text_dst = ScrolledText(frame_dst, wrap="word")
        self.text_dst.grid(row=1, column=0, sticky="nsew")
        self.text_dst.configure(font=self.font_main)
        self.text_dst.tag_configure("tentative", foreground="#999999")
        self.text_dst.tag_configure("committed", foreground="#000000")

        # 底部按钮
        bottom = tk.Frame(self.root)
        bottom.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        self.clear_btn = tk.Button(bottom, text="清屏", command=self.clear_all, font=("Segoe UI", 11))
        self.clear_btn.pack(side=tk.LEFT)
        self.quit_btn = tk.Button(bottom, text="退出", command=self.root.quit, font=("Segoe UI", 11))
        self.quit_btn.pack(side=tk.RIGHT)

        # 记录 tentative 范围
        self._tent_src_ranges = []
        self._tent_dst_ranges = []

    # ---- 下面保持不变 ----
    def ui_call(self, fn, *args, **kwargs):
        self.root.after(0, lambda: fn(*args, **kwargs))

    def set_lang_status(self, lang: str, prob: Optional[float]):
        txt = f"[lang: {lang}]" if prob is None else f"[lang: {lang}  p={prob:.2f}]"
        self.lang_label.config(text=txt)

    def set_status(self, txt: str):
        self.status_label.config(text=txt)

    def _clear_tentative(self, text_widget: ScrolledText, ranges_holder: list):
        for (start, end) in ranges_holder:
            try:
                text_widget.delete(start, end)
            except tk.TclError:
                pass
        ranges_holder.clear()

    def set_tentative_src(self, txt: str):
        self._clear_tentative(self.text_src, self._tent_src_ranges)
        if not txt:
            return
        end_index = self.text_src.index("end-1c")
        self.text_src.insert("end", txt, "tentative")
        new_end = self.text_src.index("end-1c")
        self._tent_src_ranges.append((end_index, new_end))
        self.text_src.see("end")

    def set_tentative_dst(self, txt: str):
        self._clear_tentative(self.text_dst, self._tent_dst_ranges)
        if not txt:
            return
        end_index = self.text_dst.index("end-1c")
        self.text_dst.insert("end", txt, "tentative")
        new_end = self.text_dst.index("end-1c")
        self._tent_dst_ranges.append((end_index, new_end))
        self.text_dst.see("end")

    def commit_pair(self, src: str, dst: str):
        self._clear_tentative(self.text_src, self._tent_src_ranges)
        self._clear_tentative(self.text_dst, self._tent_dst_ranges)
        if src:
            self.text_src.insert("end", src + "\n", "committed")
            self.text_src.see("end")
        if dst:
            self.text_dst.insert("end", dst + "\n", "committed")
            self.text_dst.see("end")

    def clear_all(self):
        self.text_src.delete("1.0", "end")
        self.text_dst.delete("1.0", "end")
        self._tent_src_ranges.clear()
        self._tent_dst_ranges.clear()

    def run(self):
        self.root.mainloop()




# ---------------- Translation helpers ----------------
def count_commas(text: str) -> int:
    if not text:
        return 0
    # 统计中英日常见逗号（包括日文顿号）
    return sum(ch in {',', '，', '、'} for ch in text)


def has_period(text: str) -> bool:
    if not text:
        return False
    return any(ch in {'。', '.', '！', '？', '!', '?'} for ch in text)  # 你提出重点是句号，这里也放宽问号/感叹号可选


# ---------------- Worker: Audio -> ASR -> Translate -> GUI ----------------
def worker_pipeline(args, gui: CaptionGUI, stop_event: threading.Event):
    # 初始化 ASR
    gui.ui_call(gui.set_status, f"加载 ASR 模型 {args.model} ...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    # 音频队列
    q_audio = queue.Queue()
    accumulated = np.zeros((0,), dtype=np.float32)
    samplerate = 16000
    channels = 1
    stream_ctx = None

    com_inited = False
    try:
        if args.source == "system":
            # 本线程也要初始化 COM
            pythoncom.CoInitialize()
            com_inited = True

            speaker = sc.default_speaker()
            if speaker is None:
                gui.ui_call(gui.set_status, "找不到默认扬声器，请检查系统声音设置。")
                return
            loop_mic = sc.get_microphone(speaker.name, include_loopback=True)
            samplerate = 48000
            channels = 2
            frames_per_chunk = int(samplerate * args.chunk_sec)

            def system_audio_producer():
                pythoncom.CoInitialize()
                try:
                    with loop_mic.recorder(samplerate=samplerate, channels=channels, blocksize=frames_per_chunk) as rec:
                        while not stop_event.is_set():
                            data = rec.record(numframes=frames_per_chunk)
                            mono = data.mean(axis=1).astype(np.float32, copy=False)
                            mono *= args.vol_gain
                            q_audio.put(mono)
                finally:
                    pythoncom.CoUninitialize()

            threading.Thread(target=system_audio_producer, daemon=True).start()
            gui.ui_call(gui.set_status, f"SYSTEM 回环采集中：{speaker.name} @ {samplerate}Hz")

        else:
            device_idx = find_default_input_device_index()
            if device_idx is None:
                gui.ui_call(gui.set_status, "未找到可用的麦克风输入设备。")
                return
            samplerate = get_samplerate_for_device(device_idx, fallback=16000)
            channels = 1
            blocksize = max(1, int(samplerate * args.chunk_sec))
            devname = sd.query_devices(device_idx)['name']

            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(status, file=sys.stderr)
                if indata.ndim == 2:
                    mono = indata.mean(axis=1)
                else:
                    mono = indata.squeeze(-1)
                mono = (mono * args.vol_gain).astype(np.float32, copy=False)
                q_audio.put(mono.copy())

            stream_ctx = sd.InputStream(
                device=device_idx,
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
                blocksize=blocksize,
                callback=audio_callback,
            )
            stream_ctx.__enter__()
            gui.ui_call(gui.set_status, f"MIC 采集中：{devname} @ {samplerate}Hz")

        # 状态
        last_seg = None
        src_accum = ""            # 正在累计的原文（用于发送到翻译）
        dst_accum = ""            # 正在累计的译文（未确定）
        last_feed_time = time.time()  # 最近一次把新文本送翻译的时刻
        idle_timeout_sec = 5.0        # 5 秒无新片段就终止落地
        last_tick = time.time()

        # 记录上一提交（可选用于防重复，不过我们已终止清空，不会重复）
        # last_committed_src = ""
        # last_committed_dst = ""

        while not stop_event.is_set():
            # 取音频块
            try:
                block = q_audio.get(timeout=0.2)

                # 能量门限（过滤静音/极低能量），系统回采阈值略高
                rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)) + 1e-12)
                if rms < (0.0015 if args.source == "system" else 0.0005):
                    continue

                accumulated = np.concatenate([accumulated, block])
            except queue.Empty:
                pass

            # 达到阈值：喂给 ASR
            if accumulated.shape[0] >= int(samplerate * args.accum_sec):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name
                try:
                    sf.write(wav_path, accumulated, samplerate)

                    kw = dict(
                        task="transcribe",
                        beam_size=args.beam_size,
                        patience=1.0,
                        temperature=0.0,
                        best_of=1,
                        compression_ratio_threshold=2.2,
                        log_prob_threshold=-0.5,
                        no_speech_threshold=0.2,
                        suppress_blank=True,
                        suppress_tokens=[-1],
                        condition_on_previous_text=False,
                        word_timestamps=True,
                        language=(args.lang or "ja"),
                        initial_prompt="Do not add any greetings or other non-existent text.",
                    )
                    if args.vad:
                        kw["vad_filter"] = True
                        kw["vad_parameters"] = dict(min_silence_duration_ms=args.min_silence_ms)

                    segments, info = model.transcribe(wav_path, **kw)
                    segments = list(segments)

                    # 标点化
                    lang_tag = getattr(info, "language", None)
                    text = punctuate_segments(
                        segments,
                        lang_hint=lang_tag,
                        short_pause=args.short_pause,
                        long_pause=args.long_pause,
                    )
                    text = apply_blacklist(text)

                    # 跨批次首词-末词时间差补标点（上批末词 vs 新批首词）
                    if last_seg and segments:
                        last_words = getattr(last_seg[-1], "words", None)
                        first_words = getattr(segments[0], "words", None)
                        if last_words and first_words:
                            last_word = last_words[-1]
                            first_word = first_words[0]
                            if (last_word and first_word and
                                last_word.end is not None and first_word.start is not None):
                                gap = first_word.start - last_word.end
                                if gap >= args.long_pause:
                                    text = "。" + text
                                elif gap >= args.short_pause:
                                    text = "、" + text

                    if text:
                        # 更新语言概率
                        prob = getattr(info, "language_probability", None)
                        gui.ui_call(gui.set_lang_status, (lang_tag or "auto"), prob)

                        # ==== 增量式翻译 ====
                        src_accum += text
                        gui.ui_call(gui.set_tentative_src, src_accum)

                        if args.trans_enable:
                            try:
                                res = createRequest(src_accum, args.trans_from, args.trans_to)
                                trans_text = "".join(res.get("translation", []))
                            except Exception as e:
                                trans_text = dst_accum  # 出错保留旧译文
                                print(f"[Youdao] 请求失败: {e}", file=sys.stderr)

                            dst_accum = trans_text or ""
                            gui.ui_call(gui.set_tentative_dst, dst_accum)

                            last_feed_time = time.time()

                            # ===== 终止条件判定 =====
                            # 1) 译文出现“句号”（含中英句号/问号/叹号）
                            cond_period = has_period(dst_accum)
                            # 2) 译文逗号数达到 5（中/英/顿号）
                            cond_commas = (count_commas(dst_accum) >= 5)
                            # 3) （在下面统一检查超时）

                            if cond_period or cond_commas:
                                # 终止后：先提交，再完全清空（防重复）
                                commit_src = src_accum.strip()
                                commit_dst = dst_accum.strip()
                                gui.ui_call(gui.commit_pair, commit_src, commit_dst)

                                # 清空一切累计（关键点 #1）
                                src_accum = ""
                                dst_accum = ""
                                gui.ui_call(gui.set_tentative_src, "")
                                gui.ui_call(gui.set_tentative_dst, "")

                        # 记录上一批
                        last_seg = segments

                finally:
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

                # 清空波形缓冲
                accumulated = np.zeros((0,), dtype=np.float32)

            # ===== 空闲超时终止（关键点 #2：5s 无新 segments）=====
            now = time.time()
            if (now - last_feed_time) >= 5.0:
                # 有未落地的内容就直接落地并清空
                if dst_accum.strip() or src_accum.strip():
                    gui.ui_call(gui.commit_pair, src_accum.strip(), dst_accum.strip())
                    src_accum = ""
                    dst_accum = ""
                    gui.ui_call(gui.set_tentative_src, "")
                    gui.ui_call(gui.set_tentative_dst, "")
                last_feed_time = now  # 重置计时器，避免重复触发

            # 心跳
            if time.time() - last_tick > 1.5:
                gui.ui_call(gui.set_status, f"运行中 {datetime.now().strftime('%H:%M:%S')}")
                last_tick = time.time()

    except Exception as e:
        gui.ui_call(gui.set_status, f"错误：{e}")
        raise
    finally:
        if stream_ctx is not None:
            try:
                stream_ctx.__exit__(None, None, None)
            except Exception:
                pass
        if com_inited:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass
        gui.ui_call(gui.set_status, "已停止")


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Real-time ASR (mic/system) + Pause-based punctuation + Youdao incremental translation + GUI"
    )
    parser.add_argument("--list-devices", action="store_true", help="仅列出 sounddevice 设备后退出")
    parser.add_argument("--source", choices=["mic", "system"], default="mic",
                        help="音频来源：mic=麦克风（sounddevice）；system=系统回环（soundcard）")
    parser.add_argument("--model", default="large-v3",
                        help="faster-whisper 模型：tiny/base/small/medium/large-v2/large-v3/distil-large-v3")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("--compute_type", default="float16",
                        help='计算类型：cuda 常用 "float16"；CPU 可用 "int8"/"int8_float16"')
    parser.add_argument("--beam-size", type=int, default=5, help="beam search 大小")
    parser.add_argument("--lang", default="ja", help="源语言代码（如 zh/ja/ko/en）。默认为 ja")
    parser.add_argument("--chunk-sec", type=float, default=0.6,
                        help="单次采集块大小（秒），越小越快，但 CPU 压力大")
    parser.add_argument("--accum-sec", type=float, default=1.8,
                        help="累计这么多秒就送 ASR，越小延迟越低，准确略降")
    parser.add_argument("--vad", action="store_true", help="启用内置 VAD 过滤静音")
    parser.add_argument("--min-silence-ms", type=int, default=1000, help="VAD 最小静音阈值（毫秒）")

    # 停顿阈值
    parser.add_argument("--short-pause", type=float, default=0.35, help="顿号阈值（秒）")
    parser.add_argument("--long-pause", type=float, default=0.95, help="句号阈值（秒）")

    parser.add_argument("--vol-gain", type=float, default=1.0, help="音量增益")

    # 翻译
    parser.add_argument("--trans-enable", action="store_true", default=True, help="启用翻译（默认启用）")
    parser.add_argument("--trans-from", default="ja", help="翻译源语言（有道代码，如 ja/en/zh-CHS 等）")
    parser.add_argument("--trans-to", default="zh-CHS", help="翻译目标语言（有道代码）")
    # 终止字符列表仅用于“句号判断”，5逗号/5秒超时在代码里已另行判断
    parser.add_argument("--trans-terminators", default="。.", help="用于识别句号的字符（默认：中文句号+英文句号）")

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    gui = CaptionGUI("ASR + Youdao Translation (Incremental)")
    stop_event = threading.Event()

    t = threading.Thread(target=worker_pipeline, args=(args, gui, stop_event), daemon=True)
    t.start()
    try:
        gui.run()
    finally:
        stop_event.set()
        t.join(timeout=2.0)


if __name__ == "__main__":
    main()
