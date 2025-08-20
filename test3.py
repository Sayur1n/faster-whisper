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
# ----------------------------- utils -----------------------------

def get_samplerate_for_device(device_idx: int, fallback: int = 16000) -> int:
    try:
        info = sd.query_devices(device_idx)
        sr = int(info.get("default_samplerate") or fallback)
        return sr
    except Exception:
        return fallback


def find_default_input_device_index() -> Optional[int]:
    """找到一个可用的输入设备索引（麦克风）。"""
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


def sanitize_text(text: str) -> str:
    """轻度清洗：去首尾空白。"""
    return text.strip()


def apply_blacklist(text: str) -> str:
    """
    黑名单兜底：去掉常见的“结束语”幻觉（只在这句几乎是独立短句时移除）。
    """
    t = sanitize_text(text)
    blacklist = [
        "ご視聴ありがとうございました、",
        "ご視聴ありがとうございました",
        "ご清聴ありがとうございました",
        "ありがとうございました",
        "以上です",
        "以上となります",
        "視聴ありがとうございました",
    ]
    # 只处理“几乎全句就是这些短语”的情况
    for p in blacklist:
        if t == p or (len(t) <= len(p) + 2 and p in t):
            return ""
        if any([a == p or (len(a) <= len(p) + 2 and p in a) for a in t.split('、')]):
            return ""
        if any([a == p or (len(a) <= len(p) + 2 and p in a) for a in t.split('。')]):
            return ""
    return text


def diff_append(old: str, new: str) -> str:
    """
    返回 new 中相对于 old 的新增尾部（用于只打印增量）。
    """
    if not old:
        return new
    # 寻找最大重叠
    max_overlap = min(len(old), len(new))
    for k in range(max_overlap, -1, -1):
        if old.endswith(new[:k]):
            return new[k:]
    return new


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-time audio capture (mic/system via soundcard) -> chunk -> faster-whisper transcribe with rollback"
    )
    parser.add_argument("--list-devices", action="store_true", help="仅列出 sounddevice 设备后退出（用于麦克风排查）")
    parser.add_argument("--source", choices=["mic", "system"], default="mic",
                        help="音频来源：mic=麦克风（sounddevice）；system=系统回环（soundcard）")
    parser.add_argument("--model", default="distil-large-v3",
                        help="Whisper/faster-whisper 模型名，如 tiny/base/small/medium/large-v2/large-v3/distil-large-v3")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="推理设备")
    parser.add_argument("--compute_type", default="float16",
                        help='计算类型：cuda 常用 "float16"；CPU 可用 "int8" 或 "int8_float16"')
    parser.add_argument("--beam-size", type=int, default=3, help="beam search 大小（保守一点即可）")
    parser.add_argument("--lang", default=None,
                        help="源语言代码（如 zh/ja/ko/en）。留空自动检测")
    parser.add_argument("--chunk-sec", type=float, default=0.6,
                        help="单次采集块大小（秒），越小越快，但 CPU 压力大")
    parser.add_argument("--accum-sec", type=float, default=2.0,
                        help="每个识别窗口长度（秒）")
    parser.add_argument("--overlap-sec", type=float, default=0.5,
                        help="相邻窗口重叠时长（秒），用于回滚修正")
    parser.add_argument("--stabilize-tail-sec", type=float, default=0.7,
                        help="每个窗口尾部作为“可回滚区”的时长（秒）")
    parser.add_argument("--vad", action="store_true",
                        help="启用内置 VAD 过滤静音")
    parser.add_argument("--min-silence-ms", type=int, default=800,
                        help="VAD 最小静音阈值（毫秒），仅在 --vad 时有效")
    parser.add_argument("--vol-gain", type=float, default=1.0,
                        help="音量增益（录到的 PCM 乘以该系数）")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # 初始化 ASR
    print(f"[ASR] loading model: {args.model} on {args.device} ({args.compute_type})")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    q = queue.Queue()

    # ====== 音频采集：system 用 soundcard；mic 用 sounddevice ======
    stream_ctx = None
    if args.source == "system":
        # 使用默认扬声器的 loopback
        speaker = sc.default_speaker()
        if speaker is None:
            print("找不到默认扬声器，请检查 Windows 声音设置。", file=sys.stderr)
            sys.exit(1)
        loop_mic = sc.get_microphone(speaker.name, include_loopback=True)

        samplerate = 48000  # ✅ 用 48k，降低断流概率；Whisper 内部会重采样到 16k
        channels = 2
        frames_per_chunk = int(samplerate * args.chunk_sec)

        print(f"[Audio] source=SYSTEM (soundcard loopback), device={speaker.name}, "
              f"samplerate={samplerate}, channels={channels}, chunk={args.chunk_sec}s")

        import threading

        def system_audio_producer():
            # ✅ 线程内初始化 COM，避免 0x800401f0
            pythoncom.CoInitialize()
            try:
                with loop_mic.recorder(
                    samplerate=samplerate,
                    channels=channels,
                    blocksize=frames_per_chunk,  # ✅ 显式块大小
                    # exclusive_mode=True,       # 如仍有断流可尝试独占模式
                ) as rec:
                    while True:
                        data = rec.record(numframes=frames_per_chunk)  # (frames, channels)
                        q.put(data.copy())  # ✅ 只搬运数据，处理移到主线程
            finally:
                pythoncom.CoUninitialize()

        threading.Thread(target=system_audio_producer, daemon=True).start()

    else:
        # 麦克风：sounddevice
        device_idx = find_default_input_device_index()
        if device_idx is None:
            print("未找到可用的麦克风输入设备。", file=sys.stderr)
            sys.exit(1)
        samplerate = get_samplerate_for_device(device_idx, fallback=48000)  # 用设备默认，通常 48k
        channels = 1
        blocksize = max(1, int(samplerate * args.chunk_sec))
        print(f"[Audio] source=MIC (sounddevice), device={device_idx} ({sd.query_devices(device_idx)['name']}), "
              f"samplerate={samplerate}, channels={channels}, chunk={args.chunk_sec}s")

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())  # ✅ 只入队，处理放主线程

        try:
            stream_ctx = sd.InputStream(
                device=device_idx,
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
                blocksize=blocksize,
                callback=audio_callback,
            )
        except Exception as e:
            print(f"打开麦克风输入失败：{e}", file=sys.stderr)
            sys.exit(1)

    # ====== 主循环：重叠窗口 -> 写临时 wav -> transcribe -> 回滚修正 ======
    print("[Run] Capturing audio...  Ctrl+C 退出")

    # 重叠参数
    window_frames = int(samplerate * args.accum_sec)
    overlap_frames = int(samplerate * args.overlap_sec)
    tail_cut = max(0.0, min(args.stabilize_tail_sec, args.accum_sec))  # 安全夹取

    # 音频缓冲（用 list 拼接更省拷贝）
    buffer: List[np.ndarray] = []
    buffered_frames = 0

    # 文本缓冲
    committed_text = ""  # 已“定稿”文本
    last_tail_print = ""  # 上次展示的尾部（用于覆盖演示）

    # 进入流
    entered = False
    try:
        if stream_ctx is not None:
            stream_ctx.__enter__()
            entered = True

        while True:
            block = q.get()
            # 主线程统一做：转单声道 + 增益
            if block.ndim == 2:
                mono = block.mean(axis=1)
            else:
                mono = block.squeeze(-1)
            mono = (mono * args.vol_gain).astype(np.float32, copy=False)

            buffer.append(mono)
            buffered_frames += mono.shape[0]

            # 达到窗口长度则触发识别
            if buffered_frames >= window_frames:
                # 组装一个完整窗口
                window = np.concatenate(buffer, axis=0)
                window = window[:window_frames]  # 截到窗口长度
                # 下一窗保留 overlap
                remain = window[-overlap_frames:] if overlap_frames > 0 else np.zeros((0,), dtype=np.float32)
                buffer = [remain]
                buffered_frames = remain.shape[0]

                # 写临时 wav
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name
                try:
                    sf.write(wav_path, window, samplerate)

                    # 组装保守解码参数 + 初始提示（避免结束语）
                    prompt_tail = committed_text[-200:]  # 不要太长
                    iprompt = (
                        "句読点を付けるが、挨拶や定型の締めくくり文（例：ご視聴ありがとうございました、以上です）は含めない。"
                        "不要な礼儀表現は出力しない。"
                    )

                    kw = dict(
                        beam_size=args.beam_size,
                        temperature=0.0,
                        condition_on_previous_text=True,      # ✅ 打开上下文，有助修正
                        initial_prompt=(prompt_tail + "\n" + iprompt),
                        word_timestamps=True,                  # ✅ 用词级时间做稳定尾部
                        no_speech_threshold=0.6,
                        log_prob_threshold=-0.3,
                    )
                    if args.lang:
                        kw["language"] = args.lang
                    if args.vad:
                        kw["vad_filter"] = True
                        kw["vad_parameters"] = dict(min_silence_duration_ms=args.min_silence_ms)

                    segments, info = model.transcribe(wav_path, **kw)

                    # 拼接整窗文本 + 用词时间划分“定稿/尾部”
                    words = []
                    seg_text_all = []
                    for seg in segments:
                        seg_text_all.append(seg.text)
                        if getattr(seg, "words", None):
                            words.extend(seg.words)

                    chunk_text = sanitize_text("".join(seg_text_all))
                    chunk_text = apply_blacklist(chunk_text)  # ✅ 黑名单兜底
                    if not chunk_text:
                        # 无内容则跳过，但仍保留 overlap 机制
                        continue

                    chunk_dur = window.shape[0] / samplerate
                    stable_end = max(0.0, chunk_dur - tail_cut)

                    stable_text = ""
                    tail_text = ""

                    if words:
                        # 按词时间拆分
                        stable_words = [w.word for w in words if (w.end or 0.0) <= stable_end]
                        tail_words = [w.word for w in words if (w.end or 0.0) > stable_end]
                        stable_text = sanitize_text("".join(stable_words))
                        tail_text = sanitize_text("".join(tail_words))
                    else:
                        # 没有词级时间，退化按比例切分
                        if len(chunk_text) > 0:
                            cut_idx = int(len(chunk_text) * (stable_end / max(chunk_dur, 1e-6)))
                            stable_text = sanitize_text(chunk_text[:cut_idx])
                            tail_text = sanitize_text(chunk_text[cut_idx:])

                    # 追加“定稿”部分（只打印增量）
                    inc = diff_append(committed_text, committed_text + stable_text)
                    if inc:
                        committed_text += stable_text
                        print(f"✅ {inc}", flush=True)

                    # 展示“可回滚”的尾部（覆盖上一条）
                    # 为简化，这里直接打印一条新行；你也可以用 '\r' + 宽度填充来原地覆盖
                    if tail_text != last_tail_print:
                        last_tail_print = tail_text
                        if tail_text:
                            print(f"⏳ {tail_text}", flush=True)

                finally:
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

    except KeyboardInterrupt:
        print("\n[Stop] bye.")
    finally:
        if entered:
            stream_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
