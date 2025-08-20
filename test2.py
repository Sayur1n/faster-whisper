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
from soundcard import mediafoundation as mf  # 线程内初始化/释放 COM
from faster_whisper import WhisperModel
import pythoncom


# 等价于在控制台里 set KMP_DUPLICATE_LIB_OK=TRUE
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 等价于 set HF_HOME=D:\hf_cache
os.environ["HF_HOME"] = r"D:\hf_cache"

def sanitize_text(text: str) -> str:
    return text.strip()

def apply_blacklist(text: str) -> str:
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
    for p in blacklist:
        if t == p or (len(t) <= len(p) + 2 and p in t):
            return ""
        if any([a == p or (len(a) <= len(p) + 2 and p in a) for a in t.split('、')]):
            return ""
        if any([a == p or (len(a) <= len(p) + 2 and p in a) for a in t.split('。')]):
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

# ========== 核心：基于停顿自动加标点 ==========
def punctuate_segments(
    segments,
    lang_hint: str = None,
    short_pause: float = 0.30,
    long_pause: float = 0.80,
) -> str:
    """
    先把 segments 串起来，再根据相邻 segments / words 的时间间隔决定是否加标点。
    """
    lang = (lang_hint or "").lower()
    is_cjk = any(x in lang for x in ["zh", "ja", "ko"])

    COMMA = "、" if is_cjk else ","
    PERIOD = "。" if is_cjk else "."

    out = []
    prev_end = None

    for seg in segments:
        words = getattr(seg, "words", None)
        if words:
            for w in words:
                token = (w.word or "").strip()
                if not token:
                    continue

                # 跨 segment / word 判断停顿
                if prev_end is not None and w.start is not None:
                    gap = w.start - prev_end
                    if gap >= long_pause:
                        if out and out[-1] not in f"{COMMA}{PERIOD}.,!?！？。":
                            out.append(PERIOD)
                    elif gap >= short_pause:
                        if out and out[-1] not in f"{COMMA}{PERIOD}.,!?！？。":
                            out.append(COMMA)

                out.append(token)
                prev_end = w.end or w.start
        else:
            # 没有词级时间戳，退化为段落拼接
            seg_text = (seg.text or "").strip()
            if seg_text:
                if prev_end is not None and seg.start is not None:
                    gap = seg.start - prev_end
                    if gap >= long_pause and out and out[-1] not in f"{COMMA}{PERIOD}.,!?！？。":
                        out.append(PERIOD)
                    elif gap >= short_pause and out and out[-1] not in f"{COMMA}{PERIOD}.,!?！？。":
                        out.append(COMMA)
                out.append(seg_text)
                prev_end = seg.end or prev_end

    # 最后收尾：如果没句号，可以加一个
    if out and out[-1] not in f"{PERIOD}。.!?！？":
        out.append('、')

    return "".join(out).strip()

def main():
    parser = argparse.ArgumentParser(
        description="Real-time audio capture (mic/system) -> chunk -> faster-whisper transcribe with punctuation by pauses"
    )
    parser.add_argument("--list-devices", action="store_true", help="仅列出 sounddevice 设备后退出（用于麦克风排查）")
    parser.add_argument("--source", choices=["mic", "system"], default="mic",
                        help="音频来源：mic=麦克风（sounddevice）；system=系统回环（soundcard）")
    parser.add_argument("--model", default="large-v3",
                        help="faster-whisper 模型：tiny/base/small/medium/large-v2/large-v3/distil-large-v3")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("--compute_type", default="float16",
                        help='计算类型：cuda 常用 "float16"；CPU 可用 "int8"/"int8_float16"')
    parser.add_argument("--beam-size", type=int, default=5, help="beam search 大小")
    parser.add_argument("--lang", default=None, help="源语言代码（如 zh/ja/ko/en）。留空自动检测")
    parser.add_argument("--chunk-sec", type=float, default=0.6,
                        help="单次采集块大小（秒），越小越快，但 CPU 压力大")
    parser.add_argument("--accum-sec", type=float, default=2.0,
                        help="累计这么多秒就送 ASR，越小延迟越低，准确略降")
    parser.add_argument("--vad", action="store_true", help="启用内置 VAD 过滤静音")
    parser.add_argument("--min-silence-ms", type=int, default=500, help="VAD 最小静音阈值（毫秒）")

    # 新增：停顿阈值（可按语速调整）
    parser.add_argument("--short-pause", type=float, default=0.30, help="顿号阈值（秒）")
    parser.add_argument("--long-pause", type=float, default=0.80, help="句号阈值（秒）")

    parser.add_argument("--vol-gain", type=float, default=1.0, help="音量增益")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # ASR 初始化
    print(f"[ASR] loading model: {args.model} on {args.device} ({args.compute_type})")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    q = queue.Queue()
    accumulated = np.zeros((0,), dtype=np.float32)

    # ====== 音频采集设置 ======
    stream_ctx = None
    if args.source == "system":
        speaker = sc.default_speaker()
        if speaker is None:
            print("找不到默认扬声器，请检查 Windows 声音设置。", file=sys.stderr)
            sys.exit(1)
        loop_mic = sc.get_microphone(speaker.name, include_loopback=True)

        samplerate = 48000
        channels = 2
        frames_per_chunk = int(samplerate * args.chunk_sec)

        print(f"[Audio] source=SYSTEM (loopback), device={speaker.name}, "
              f"samplerate={samplerate}, channels={channels}")

        import threading
        def system_audio_producer():
            pythoncom.CoInitialize()
            try:
                with loop_mic.recorder(samplerate=samplerate, channels=channels, blocksize=frames_per_chunk) as rec:
                    while True:
                        data = rec.record(numframes=frames_per_chunk)
                        mono = data.mean(axis=1).astype(np.float32, copy=False)
                        mono *= args.vol_gain
                        q.put(mono)
            finally:
                pythoncom.CoUninitialize()

        threading.Thread(target=system_audio_producer, daemon=True).start()

    else:
        device_idx = find_default_input_device_index()
        if device_idx is None:
            print("未找到可用的麦克风输入设备。", file=sys.stderr)
            sys.exit(1)
        samplerate = get_samplerate_for_device(device_idx, fallback=16000)
        channels = 1
        blocksize = max(1, int(samplerate * args.chunk_sec))
        devname = sd.query_devices(device_idx)['name']
        print(f"[Audio] source=MIC, device={device_idx} ({devname}), samplerate={samplerate}, channels={channels}")

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            if indata.ndim == 2:
                mono = indata.mean(axis=1)
            else:
                mono = indata.squeeze(-1)
            mono = (mono * args.vol_gain).astype(np.float32, copy=False)
            q.put(mono.copy())

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

    # ====== 主循环 ======
    print("[Run] Capturing audio...  Ctrl+C 退出")
    entered = False
    try:
        if stream_ctx is not None:
            stream_ctx.__enter__()
            entered = True

        last_tick = time.time()
        last_seg = None
        while True:
            block = q.get()
            accumulated = np.concatenate([accumulated, block])

            if accumulated.shape[0] >= int(samplerate * args.accum_sec):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name
                try:
                    sf.write(wav_path, accumulated, samplerate)

                    kw = dict(
                        beam_size=args.beam_size,
                        temperature=0.0,                # 更稳定
                        condition_on_previous_text=False, # 按你的要求保持关闭
                        word_timestamps=True,            # 关键：开启词级时间戳
                    )
                    if args.lang:
                        kw["language"] = args.lang
                    if args.vad:
                        kw["vad_filter"] = True
                        kw["vad_parameters"] = dict(min_silence_duration_ms=args.min_silence_ms)

                    segments, info = model.transcribe(wav_path, **kw)
                    segments = list(segments)
                    # 用停顿时长打标点
                    lang_tag = getattr(info, "language", None)
                    text = punctuate_segments(
                        segments,
                        lang_hint=lang_tag,
                        short_pause=args.short_pause,
                        long_pause=args.long_pause,
                    )
                    # 黑名单兜底
                    text = apply_blacklist(text)

                    # 处理segments间的标点
                    COMMA = "、" 
                    PERIOD = "。"
                    if last_seg:
                        last_words = getattr(last_seg[-1], "words", None)
                        last_word = last_words[-1]
                        if segments:
                            words = getattr(segments[0], "words", None)
                            word = words[-1]
                            print(f"上一个:{last_word.word}.{last_word.end}")
                            print(f"下一个:{word.word}.{word.start}")
                            if word.start - last_word.end >= args.long_pause:
                                text = PERIOD + text
                            elif word.start - last_word.end >= args.short_pause:
                                text = COMMA + text

                    if text:
                        lang = (lang_tag or "auto")
                        prob = getattr(info, "language_probability", None)
                        head = f"[{lang}]" if prob is None else f"[{lang} p={prob:.2f}]"
                        print(f"{head} {text}")

                    # 状态机
                    last_seg = segments

                finally:
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

                accumulated = np.zeros((0,), dtype=np.float32)

            if time.time() - last_tick > 0.1:
                last_tick = time.time()

    except KeyboardInterrupt:
        print("\n[Stop] bye.")
    finally:
        if entered:
            stream_ctx.__exit__(None, None, None)

if __name__ == "__main__":
    main()
