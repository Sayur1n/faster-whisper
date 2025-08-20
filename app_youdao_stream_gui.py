# -*- coding: utf-8 -*-
"""
Real-time Speech Translation (Youdao streaming API) + GUI
- 采集音频（mic/system）
- 重采样到 16 kHz 单声道，编码为 PCM16
- 通过有道「流式语音翻译」WebSocket 发送
- 流式显示：partial(灰色覆盖) → final(黑色落地)
- 空闲 5s 自动落地清空
- 可选 --debug-partial 打印/落盘分句增量行为

依赖（来自你的 demo 目录）：
- apidemo_2/StreamSpeechTransDemo.py: init_connection_with_params, send_binary_message
- apidemo_2/utils/AuthV3Util.py: addAuthParams
"""

import argparse
import os
import sys
import time
import queue
import json
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import soundcard as sc
from tkinter.scrolledtext import ScrolledText
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
import pythoncom  # Windows COM（system loopback 需要在线程内初始化）

# ====== 引入你提供的有道 Demo 辅助函数 ======
# 注意：确保 Python 的工作目录/模块路径能找到 apidemo_2
from apidemo_2.StreamSpeechTransDemo import init_connection_with_params, send_binary_message
from apidemo_2.utils.AuthV3Util import addAuthParams


# ---------------- Utils: devices & audio ----------------
def build_wav_header(sample_rate=16000, bits_per_sample=16, channels=1, data_bytes_hint=0x7fffffff):
    """
    生成标准 44 字节 WAV 头（PCM、LE）。data_bytes_hint 用一个大占位值即可，流式不严格校验。
    """
    import struct
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    riff_chunk_size = 36 + data_bytes_hint  # RIFF chunk size = 36 + data_bytes
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_chunk_size,
        b"WAVE",
        b"fmt ",
        16,                     # PCM 子块大小
        1,                      # PCM 格式
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_bytes_hint
    )
    return header


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


def resample_mono(x: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    """线性插值重采样到 dst_sr，输入/输出均为 mono float32（-1~1）。"""
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    t_src = np.linspace(0.0, 1.0, num=x.size, endpoint=False, dtype=np.float64)
    n_dst = int(round(x.size * (dst_sr / float(src_sr))))
    if n_dst <= 0:
        n_dst = 1
    t_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False, dtype=np.float64)
    y = np.interp(t_dst, t_src, x.astype(np.float64))
    return y.astype(np.float32, copy=False)


def float32_to_int16_pcm(x: np.ndarray) -> bytes:
    """[-1,1] float32 → int16 PCM little-endian 原始字节。"""
    y = np.clip(x, -1.0, 1.0)
    y = (y * 32767.0).astype(np.int16)
    return y.tobytes()


# ---------------- GUI ----------------
class CaptionGUI:
    """
    上下布局：
    - 上：原文 (from API)
    - 下：译文 (Youdao streaming translation)
    窗口 600x400，上下等高，文字过多时自动下翻
    """
    def __init__(self, title="Youdao Streaming Speech Translation"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("600x400")

        self.font_main = tkfont.Font(family="Segoe UI", size=16)

        # 行列权重
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.columnconfigure(0, weight=1)

        # 顶部状态栏
        top = tk.Frame(self.root)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        self.lang_label = tk.Label(top, text="[from: ?  to: ?]", anchor="w", font=("Segoe UI", 11))
        self.lang_label.pack(side=tk.LEFT)
        self.status_label = tk.Label(top, text="准备就绪", anchor="e", font=("Segoe UI", 11))
        self.status_label.pack(side=tk.RIGHT)

        # 原文
        frame_src = tk.Frame(self.root)
        frame_src.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        frame_src.rowconfigure(1, weight=1)
        frame_src.columnconfigure(0, weight=1)
        tk.Label(frame_src, text="原文 (from API)", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 2)
        )
        self.text_src = ScrolledText(frame_src, wrap="word")
        self.text_src.grid(row=1, column=0, sticky="nsew")
        self.text_src.configure(font=self.font_main)
        self.text_src.tag_configure("tentative", foreground="#999999")
        self.text_src.tag_configure("committed", foreground="#000000")

        # 译文
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

        # 底部
        bottom = tk.Frame(self.root)
        bottom.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        self.clear_btn = tk.Button(bottom, text="清屏", command=self.clear_all, font=("Segoe UI", 11))
        self.clear_btn.pack(side=tk.LEFT)
        self.quit_btn = tk.Button(bottom, text="退出", command=self.root.quit, font=("Segoe UI", 11))
        self.quit_btn.pack(side=tk.RIGHT)

        # 记录当前灰色区域范围，方便覆盖更新
        self._tent_src_ranges = []
        self._tent_dst_ranges = []

    def ui_call(self, fn, *args, **kwargs):
        self.root.after(0, lambda: fn(*args, **kwargs))

    def set_lang_status(self, lang_from: str, lang_to: str):
        txt = f"[from: {lang_from}  to: {lang_to}]"
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
        start_idx = self.text_src.index("end-1c")
        self.text_src.insert("end", txt, "tentative")
        end_idx = self.text_src.index("end-1c")
        self._tent_src_ranges.append((start_idx, end_idx))
        self.text_src.see("end")

    def set_tentative_dst(self, txt: str):
        self._clear_tentative(self.text_dst, self._tent_dst_ranges)
        if not txt:
            return
        start_idx = self.text_dst.index("end-1c")
        self.text_dst.insert("end", txt, "tentative")
        end_idx = self.text_dst.index("end-1c")
        self._tent_dst_ranges.append((start_idx, end_idx))
        self.text_dst.see("end")

    def commit_pair(self, src: str, dst: str):
        # 先清掉灰色，避免残留
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


# ---------------- Debug helpers (optional) ----------------
def _lcp_len(a: str, b: str) -> int:
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


def _short(s: str, n: int = 40) -> str:
    s = s.replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s


class PartialDebugger:
    """记录每个 segId 上一条内容，用于判断 partial 行为（append/overwrite/rewrite）"""
    def __init__(self, enable: bool, to_file: bool = True, path: str = "youdao_debug.log"):
        self.enable = enable
        self.prev = {}   # segId -> {"src": str, "dst": str}
        self.fp = None
        if enable and to_file:
            try:
                self.fp = open(path, "a", encoding="utf-8")
                self._log(f"=== START {datetime.now()} ===")
            except Exception:
                self.fp = None

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] {msg}"
        print(line)
        if self.fp:
            try:
                self.fp.write(line + "\n")
                self.fp.flush()
            except Exception:
                pass

    def log_message(self, seg_id, partial: bool, src: str, dst: str):
        if not self.enable or seg_id is None:
            return
        prev = self.prev.get(seg_id, {"src": "", "dst": ""})

        # 原文行为
        old_s, new_s = prev["src"], src or ""
        lcp_s = _lcp_len(old_s, new_s)
        if new_s.startswith(old_s):
            s_behavior = "append" if len(new_s) > len(old_s) else "same"
        elif old_s.startswith(new_s):
            s_behavior = "shrink"
        elif lcp_s > 0:
            s_behavior = f"rewrite(lcp={lcp_s})"
        else:
            s_behavior = "replace"

        # 译文行为
        old_t, new_t = prev["dst"], dst or ""
        lcp_t = _lcp_len(old_t, new_t)
        if new_t.startswith(old_t):
            t_behavior = "append" if len(new_t) > len(old_t) else "same"
        elif old_t.startswith(new_t):
            t_behavior = "shrink"
        elif lcp_t > 0:
            t_behavior = f"rewrite(lcp={lcp_t})"
        else:
            t_behavior = "replace"

        self._log(
            f"seg={seg_id} partial={partial} "
            f"src(len {len(old_s)}→{len(new_s)} {s_behavior}): “{_short(new_s)}”  |  "
            f"dst(len {len(old_t)}→{len(new_t)} {t_behavior}): “{_short(new_t)}”"
        )
        self.prev[seg_id] = {"src": new_s, "dst": new_t}

    def close(self):
        if self.fp:
            try:
                self.fp.close()
            except Exception:
                pass
            self.fp = None


# ---------------- Message parsing (Youdao WS) ----------------
def parse_youdao_message(msg: str):
    """
    解析有道流式返回消息（text message）：
      - 顶层：errorCode, action
      - result：context(原文)、tranContent(译文)、partial(是否中间结果)、segId,bg,ed
    返回: (src_text, dst_text, is_final, action, error_code, seg_id)
    """
    try:
        data = json.loads(msg)
    except Exception:
        return None, None, False, None, None, None

    error_code = str(data.get("errorCode", ""))
    action = data.get("action", "")

    # 错误或非识别阶段
    if action == "error" or (error_code and error_code != "0"):
        return None, None, False, action, error_code, None

    res = data.get("result") or {}
    src_text = res.get("context") or ""
    dst_text = res.get("tranContent") or ""
    seg_id = res.get("segId")
    is_partial = bool(res.get("partial", False))
    is_final = (not is_partial)

    return src_text, dst_text, is_final, action, error_code, seg_id


# ---------------- Worker: Audio -> Youdao WS -> GUI ----------------
def worker_pipeline(args, gui: CaptionGUI, stop_event: threading.Event):
    """
    采集音频 → 重采样到16k mono → 以PCM块(约0.2s)通过有道WebSocket流式发送
     ← WebSocket回调拿到增量/最终翻译 → 在GUI显示(partial=灰色覆盖, final=黑色落地)
    """
    gui.ui_call(gui.set_status, "连接有道流式语音翻译服务中...")
    url = "wss://openapi.youdao.com/stream_speech_trans"

    # 16k/mono/wav，且首包发送 WAV 头
    rate = "16000"
    fmt = "wav"
    params = {
        "from": args.trans_from,
        "to": args.trans_to,
        "format": fmt,
        "channel": "1",
        "version": "v1",
        "rate": rate,
        "q": ""  # AuthV3 需要，流式场景留空
    }
    addAuthParams(args.yd_app_key, args.yd_app_secret, params)
    ws_client = init_connection_with_params(url, params)

    # 等待连接
    t0 = time.time()
    while not ws_client.return_is_connect():
        if stop_event.is_set():
            gui.ui_call(gui.set_status, "已停止")
            return
        if time.time() - t0 > 10:
            gui.ui_call(gui.set_status, "连接超时，请检查 AppKey/网络")
            return
        time.sleep(0.1)

    gui.ui_call(gui.set_status, "连接成功，开始采集音频并发送...")
    gui.ui_call(gui.set_lang_status, args.trans_from, args.trans_to)

    # 调试器
    dbg = PartialDebugger(enable=getattr(args, "debug_partial", False))

    # 流式显示所需状态（同一活跃 segId 的最新假设）
    last_feed_time = time.time()
    idle_timeout_sec = 5.0
    active_seg_id = None
    active_src = ""
    active_dst = ""

    def on_message(_ws, message: str):
        nonlocal last_feed_time, active_seg_id, active_src, active_dst
        src_text, dst_text, is_final, action, err, seg_id = parse_youdao_message(message)

        if action == "started":
            gui.ui_call(gui.set_status, "已握手（started）")
            return

        if action == "error" or (err and err != "0"):
            gui.ui_call(gui.set_status, f"API错误: errorCode={err}")
            return

        if action != "recognition" or seg_id is None:
            return

        # 调试：记录 partial 行为
        if args.debug_partial:
            is_partial = not is_final
            dbg.log_message(seg_id, partial=is_partial, src=src_text or "", dst=dst_text or "")

        # 新句开始：切换活跃 seg，清空灰色
        if (active_seg_id is None) or (seg_id > active_seg_id):
            active_seg_id = seg_id
            active_src = ""
            active_dst = ""
            gui.ui_call(gui.set_tentative_src, "")
            gui.ui_call(gui.set_tentative_dst, "")

        # 仅处理当前活跃 seg
        if seg_id < active_seg_id:
            return

        # partial 通常是“最新全量假设”，用覆盖而非累加
        if src_text is not None:
            active_src = src_text
        if dst_text is not None:
            active_dst = dst_text

        if not hasattr(on_message, "_prev_src"):
            on_message._prev_src, on_message._prev_dst = "", ""

        if not is_final:
            # 仅在内容变化时刷新灰色，避免每条 partial 都刷屏
            if active_src != on_message._prev_src:
                gui.ui_call(gui.set_tentative_src, active_src)
                on_message._prev_src = active_src
            if active_dst != on_message._prev_dst:
                gui.ui_call(gui.set_tentative_dst, active_dst)
                on_message._prev_dst = active_dst
        else:
            # 最终版落地为黑色，并清空灰色与缓冲
            if active_src.strip() or active_dst.strip():
                gui.ui_call(gui.commit_pair, active_src.strip(), active_dst.strip())
            active_src = ""
            active_dst = ""
            gui.ui_call(gui.set_tentative_src, "")
            gui.ui_call(gui.set_tentative_dst, "")
            on_message._prev_src = ""
            on_message._prev_dst = ""

        last_feed_time = time.time()

    def on_error(_ws, error):
        gui.ui_call(gui.set_status, f"WS错误: {error}")

    def on_close(_ws, code, reason):
        gui.ui_call(gui.set_status, f"WS关闭: code={code}, reason={reason}")

    # 注册回调
    if hasattr(ws_client, "set_on_message"):
        ws_client.set_on_message(on_message)
        if hasattr(ws_client, "set_on_error"):
            ws_client.set_on_error(on_error)
        if hasattr(ws_client, "set_on_close"):
            ws_client.set_on_close(on_close)
    elif hasattr(ws_client, "ws"):
        try:
            ws_client.ws.on_message = on_message
        except Exception:
            pass

    # 音频采集：送 16k mono PCM
    q_audio = queue.Queue()
    stream_ctx = None
    com_inited = False

    try:
        if args.source == "system":
            pythoncom.CoInitialize()
            com_inited = True

            speaker = sc.default_speaker()
            if speaker is None:
                gui.ui_call(gui.set_status, "找不到默认扬声器")
                return
            loop_mic = sc.get_microphone(speaker.name, include_loopback=True)

            src_sr = 48000
            src_ch = 2
            frames_per_chunk = int(src_sr * args.chunk_sec)

            def system_audio_producer():
                pythoncom.CoInitialize()
                try:
                    with loop_mic.recorder(samplerate=src_sr, channels=src_ch, blocksize=frames_per_chunk) as rec:
                        while not stop_event.is_set():
                            data = rec.record(numframes=frames_per_chunk)  # (N,2)
                            mono = data.mean(axis=1).astype(np.float32, copy=False)
                            mono *= args.vol_gain
                            mono16 = resample_mono(mono, src_sr, 16000)
                            q_audio.put(mono16)
                finally:
                    pythoncom.CoUninitialize()

            threading.Thread(target=system_audio_producer, daemon=True).start()
            gui.ui_call(gui.set_status, f"SYSTEM 回采：{speaker.name} → Youdao 流式翻译")

        else:
            device_idx = find_default_input_device_index()
            if device_idx is None:
                gui.ui_call(gui.set_status, "未找到可用麦克风")
                return
            cap_sr = get_samplerate_for_device(device_idx, fallback=16000)
            cap_ch = 1
            blocksize = max(1, int(cap_sr * args.chunk_sec))
            devname = sd.query_devices(device_idx)['name']

            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(status, file=sys.stderr)
                if indata.ndim == 2:
                    mono = indata.mean(axis=1)
                else:
                    mono = indata.squeeze(-1)
                mono = (mono * args.vol_gain).astype(np.float32, copy=False)
                mono16 = resample_mono(mono, cap_sr, 16000)
                q_audio.put(mono16)

            stream_ctx = sd.InputStream(
                device=device_idx,
                samplerate=cap_sr,
                channels=cap_ch,
                dtype="float32",
                blocksize=blocksize,
                callback=audio_callback,
            )
            stream_ctx.__enter__()
            gui.ui_call(gui.set_status, f"MIC 采集：{devname} @ {cap_sr}Hz → Youdao 流式翻译")

        # 发送循环：每 ~0.2s 发一包
        chunk_ms = 200
        samples_per_send = int(16000 * (chunk_ms / 1000.0))
        buf = np.zeros((0,), dtype=np.float32)

        # 首包 WAV 头
        wav_header = build_wav_header(sample_rate=16000, bits_per_sample=16, channels=1)
        send_binary_message(ws_client.ws, wav_header)

        last_tick = time.time()
        while not stop_event.is_set():
            try:
                mono16 = q_audio.get(timeout=0.1)
                buf = np.concatenate([buf, mono16])
            except queue.Empty:
                pass

            while buf.shape[0] >= samples_per_send:
                send_chunk = buf[:samples_per_send]
                buf = buf[samples_per_send:]
                pcm_bytes = float32_to_int16_pcm(send_chunk)
                try:
                    send_binary_message(ws_client.ws, pcm_bytes)
                except Exception as e:
                    gui.ui_call(gui.set_status, f"发送失败: {e}")
                    stop_event.set()
                    break
                time.sleep(0.1)

            # 空闲超时：5s 无新结果 → 将当前灰色最新假设落地清空
            now = time.time()
            if (now - last_feed_time) >= idle_timeout_sec:
                if (active_src.strip() or active_dst.strip()):
                    gui.ui_call(gui.commit_pair, active_src.strip(), active_dst.strip())
                    active_src = ""
                    active_dst = ""
                    gui.ui_call(gui.set_tentative_src, "")
                    gui.ui_call(gui.set_tentative_dst, "")
                last_feed_time = now

            if time.time() - last_tick > 1.5:
                gui.ui_call(gui.set_status, f"运行中 {datetime.now().strftime('%H:%M:%S')}")
                last_tick = time.time()

        # 结束：通知服务端
        try:
            send_binary_message(ws_client.ws, '{"end":"true"}')
        except Exception:
            pass

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
        dbg.close()
        gui.ui_call(gui.set_status, "已停止")


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Youdao Streaming Speech Translation + GUI")
    parser.add_argument("--list-devices", action="store_true", help="仅列出 sounddevice 设备后退出")
    parser.add_argument("--source", choices=["mic", "system"], default="mic",
                        help="音频来源：mic=麦克风；system=系统回环")
    parser.add_argument("--chunk-sec", type=float, default=0.1,
                        help="采集块大小（秒），越小越快，但 CPU 压力大")
    parser.add_argument("--vol-gain", type=float, default=1.0, help="音量增益")

    # 有道鉴权
    parser.add_argument("--yd-app-key", required=True, help="Youdao APP KEY")
    parser.add_argument("--yd-app-secret", required=True, help="Youdao APP SECRET")

    # 翻译语种（按有道文档编码）
    parser.add_argument("--trans-from", default="ja", help="源语言（如 ja/en/zh-CHS 等）")
    parser.add_argument("--trans-to", default="zh-CHS", help="目标语言（如 zh-CHS/ja/en 等）")

    # 调试：观察 partial 行为
    parser.add_argument("--debug-partial", action="store_true",
                        help="打印分句 partial 行为的调试日志")

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    gui = CaptionGUI("ASR-free: Youdao Streaming Translation")
    gui.set_lang_status(args.trans_from, args.trans_to)
    stop_event = threading.Event()

    t = threading.Thread(target=worker_pipeline, args=(args, gui, stop_event), daemon=True)
    t.start()
    try:
        gui.run()
    finally:
        stop_event.set()
        t.join(timeout=2.0)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("HF_HOME", r"D:\hf_cache")
    main()
