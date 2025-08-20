from faster_whisper import WhisperModel

# 选择模型大小：tiny, base, small, medium, large-v2, large-v3, distil-large-v3
model_size = "large-v2"

# 初始化模型（GPU 上 float16 推理最快，CPU 上建议 int8）
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# 转录已有音频文件
segments, info = model.transcribe("录音.wav", beam_size=5)

print("Detected language '%s' (p=%.2f)" % (info.language, info.language_probability))

for seg in segments:  # 注意 segments 是生成器
    print(f"[{seg.start:.2f}s → {seg.end:.2f}s] {seg.text}")
