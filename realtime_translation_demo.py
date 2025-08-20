# -*- coding: utf-8 -*-
"""
Real-time Speech Translation (Youdao streaming API) + GUI
- 采集音频（mic/system）
- 重采样到 16kHz 单声道，编码为 PCM16
- 通过有道「流式语音翻译」WebSocket 发送
- 实时接收增量/最终翻译，原文/译文上下显示
- 终止就清空（防重复），空闲 5s 自动落地清空

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
    # 用简单的线性插值（足够实时、无额外依赖）
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
    - 上：原文 (ASR-like text from API)
    - 下：译文 (Youdao streaming translation)
    窗口 600x400，上下等高，文字过多时自动下翻
    """
    def __init__(self, title="Youdao Streaming Speech Translation"):
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
        self.lang_label = tk.Label(top, text="[from: ?  to: ?]", anchor="w", font=("Segoe UI", 11))
        self.lang_label.pack(side=tk.LEFT)
        self.status_label = tk.Label(top, text="准备就绪", anchor="e", font=("Segoe UI", 11))
        self.status_label.pack(side=tk.RIGHT)

        # 原文 frame
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

        # 状态缓存
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


# ---------------- Message parsing (Youdao WS) ----------------
def parse_youdao_message(msg: str):
    """
    尽量兼容地解析有道流式返回消息：
    期望包含：
      - 原文：可能在 "src", "asrText", "text" 等字段（增量/最终）
      - 译文：可能在 "trans", "translation", "dst" 等字段
      - 终止标记：可能在 "end", "isFinal", "resultType" == "final"
    返回: (src_text, dst_text, is_final)
    """
    try:
        data = json.loads(msg)
    except Exception:
        return None, None, False

    # 原文候选
    src_candidates = [
        data.get("src"),
        data.get("asrText"),
        data.get("text"),
        data.get("result", {}).get("src"),
        data.get("result", {}).get("asrText"),
        data.get("result", {}).get("text"),
    ]
    src_text = next((s for s in src_candidates if isinstance(s, str) and s.strip()), "")

    # 译文候选
    dst_candidates = [
        data.get("trans"),
        data.get("translation"),
        data.get("dst"),
        data.get("result", {}).get("trans"),
        data.get("result", {}).get("translation"),
        data.get("result", {}).get("dst"),
    ]
    # 有的接口可能给列表
    dst_text = None
    for c in dst_candidates:
        if isinstance(c, list):
            c = "".join([str(x) for x in c])
        if isinstance(c, str) and c.strip():
            dst_text = c
            break
    if dst_text is None:
        dst_text = ""

    # 是否最后一条
    is_final = False
    if str(data.get("end", "")).lower() == "true":
        is_final = True
    if str(data.get("isFinal", "")).lower() == "true":
        is_final = True
    if data.get("resultType", "") == "final":
        is_final = True

    return src_text, dst_text, is_final


# ---------------- Worker: Audio -> Youdao WS -> GUI ----------------
def worker_pipeline(args, gui: CaptionGUI, stop_event: threading.Event):
    """
    采集音频 → 重采样到16k mono → 以PCM块(约0.2s)通过有道WebSocket流式发送
     ← WebSocket回调拿到增量/最终翻译 → 在GUI显示(原文/译文)
    """
    # ========== WebSocket 连接 ==========
    gui.ui_call(gui.set_status, "连接有道流式语音翻译服务中...")
    url = "wss://openapi.youdao.com/stream_speech_trans"

    # 有道推荐：16k、单声道、pcm/wav；这里用 pcm 更适合流式（无需WAV头）
    rate = "16000"
    fmt = "pcm"
    params = {
        "from": args.trans_from,
        "to": args.trans_to,
        "format": fmt,
        "channel": "1",
        "version": "v1",
        "rate": rate,
    }
    addAuthParams(args.yd_app_key, args.yd_app_secret, params)

    ws_client = init_connection_with_params(url, params)

    # 等待连接ready
    t0 = time.time()
    while not ws_client.return_is_connect():
        if stop_event.is_set():
            gui.ui_call(gui.set_status, "已停止")
            return
        if time.time() - t0 > 10:
            gui.ui_call(gui.set_status, "连接超时，请检查 AppKey/网络")
            return
        time.sleep(0.05)

    gui.ui_call(gui.set_status, "连接成功，开始采集音频并发送...")
    gui.ui_call(gui.set_lang_status, args.trans_from, args.trans_to)

    # ========== 设置 WS 回调，接收翻译结果 ==========
    src_accum = ""  # 未提交的原文
    dst_accum = ""  # 未提交的译文
    last_feed_time = time.time()
    idle_timeout_sec = 5.0

    # 不同 demo 的客户端封装可能不同，这里兼容两种：set_on_message 或直接 ws.on_message
    def on_message(_ws, message: str):
        nonlocal src_accum, dst_accum, last_feed_time
        src_text, dst_text, is_final = parse_youdao_message(message)
        if src_text:
            src_accum += src_text
            gui.ui_call(gui.set_tentative_src, src_accum)
        if dst_text:
            dst_accum += dst_text
            gui.ui_call(gui.set_tentative_dst, dst_accum)

        last_feed_time = time.time()

        # 如果该消息标注为最终结果，那么提交并清空
        if is_final and (src_accum.strip() or dst_accum.strip()):
            gui.ui_call(gui.commit_pair, src_accum.strip(), dst_accum.strip())
            src_accum = ""
            dst_accum = ""
            gui.ui_call(gui.set_tentative_src, "")
            gui.ui_call(gui.set_tentative_dst, "")

    def on_error(_ws, error):
        gui.ui_call(gui.set_status, f"WS错误: {error}")

    def on_close(_ws, code, reason):
        gui.ui_call(gui.set_status, f"WS关闭: code={code}, reason={reason}")

    # 尝试注册
    if hasattr(ws_client, "set_on_message"):
        ws_client.set_on_message(on_message)
        if hasattr(ws_client, "set_on_error"):
            ws_client.set_on_error(on_error)
        if hasattr(ws_client, "set_on_close"):
            ws_client.set_on_close(on_close)
    elif hasattr(ws_client, "ws"):
        # 可能是 WebSocketApp 风格
        try:
            ws_client.ws.on_message = on_message
        except Exception:
            pass

    # ========== 音频采集（与原来类似，但改成直送WS） ==========
    q_audio = queue.Queue()
    stream_ctx = None
    com_inited = False

    try:
        if args.source == "system":
            pythoncom.CoInitialize()  # 本线程COM
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
                            # 重采样到16k
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
                # 重采样到16k
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

        # ========== 发送循环 ==========
        # 每次从队列中取出音频，拼成约 0.2 秒的 PCM 块发送（与官方 demo 的 step/time.sleep 保持接近）
        chunk_ms = 200
        samples_per_send = int(16000 * (chunk_ms / 1000.0))
        buf = np.zeros((0,), dtype=np.float32)

        last_tick = time.time()

        while not stop_event.is_set():
            # 收音拼块
            try:
                mono16 = q_audio.get(timeout=0.1)
                buf = np.concatenate([buf, mono16])
            except queue.Empty:
                pass

            # 够一包就发
            while buf.shape[0] >= samples_per_send:
                send_chunk = buf[:samples_per_send]
                buf = buf[samples_per_send:]

                pcm_bytes = float32_to_int16_pcm(send_chunk)
                try:
                    # 直接发送二进制 PCM
                    send_binary_message(ws_client.ws, pcm_bytes)
                except Exception as e:
                    gui.ui_call(gui.set_status, f"发送失败: {e}")
                    stop_event.set()
                    break

                # 与官方 demo 的发送节奏类似
                time.sleep(0.2)

            # 空闲超时：若 5s 无返回增量/最终，就提交并清空（避免积压）
            now = time.time()
            if (now - last_feed_time) >= idle_timeout_sec:
                if src_accum.strip() or dst_accum.strip():
                    gui.ui_call(gui.commit_pair, src_accum.strip(), dst_accum.strip())
                    src_accum = ""
                    dst_accum = ""
                    gui.ui_call(gui.set_tentative_src, "")
                    gui.ui_call(gui.set_tentative_dst, "")
                last_feed_time = now

            # 心跳
            if time.time() - last_tick > 1.5:
                gui.ui_call(gui.set_status, f"运行中 {datetime.now().strftime('%H:%M:%S')}")
                last_tick = time.time()

        # 结束：告诉服务端没有后续数据
        try:
            end_message = "{\"end\": \"true\"}"
            send_binary_message(ws_client.ws, end_message)
        except Exception:
            pass

    except Exception as e:
        gui.ui_call(gui.set_status, f"错误：{e}")
        raise
    finally:
        # 关闭 stream
        if stream_ctx is not None:
            try:
                stream_ctx.__exit__(None, None, None)
            except Exception:
                pass
        # 释放 COM
        if com_inited:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass
        gui.ui_call(gui.set_status, "已停止")


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Youdao Streaming Speech Translation + GUI"
    )
    parser.add_argument("--list-devices", action="store_true", help="仅列出 sounddevice 设备后退出")
    parser.add_argument("--source", choices=["mic", "system"], default="mic",
                        help="音频来源：mic=麦克风（sounddevice）；system=系统回环（soundcard）")
    parser.add_argument("--chunk-sec", type=float, default=0.6,
                        help="采集块大小（秒），越小越快，但 CPU 压力大")
    parser.add_argument("--vol-gain", type=float, default=1.0, help="音量增益")

    # 有道鉴权
    parser.add_argument("--yd-app-key", required=True, help="Youdao APP KEY")
    parser.add_argument("--yd-app-secret", required=True, help="Youdao APP SECRET")

    # 翻译语种（按有道文档编码）
    parser.add_argument("--trans-from", default="ja", help="源语言（如 ja/en/zh-CHS 等）")
    parser.add_argument("--trans-to", default="zh-CHS", help="目标语言（如 zh-CHS/ja/en 等）")

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
    # 推荐：避免 MKL 与缓存冲突
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("HF_HOME", r"D:\hf_cache")
    main()
