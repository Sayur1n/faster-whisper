[![CI](https://github.com/SYSTRAN/faster-whisper/workflows/CI/badge.svg)](https://github.com/SYSTRAN/faster-whisper/actions?query=workflow%3ACI) [![PyPI version](https://badge.fury.io/py/faster-whisper.svg)](https://badge.fury.io/py/faster-whisper)

# Faster Whisper 语音转录与实时翻译系统

**faster-whisper** 是基于 [CTranslate2](https://github.com/OpenNMT/CTranslate2/) 重新实现的 OpenAI Whisper 模型，CTranslate2 是一个用于 Transformer 模型的快速推理引擎。

本实现比 [openai/whisper](https://github.com/openai/whisper) 快达4倍，同时保持相同精度并减少内存使用。通过8位量化可以在 CPU 和 GPU 上进一步提升效率。

## 其他项目链接

本项目作者的其他AI相关项目：

- **🤖 AI导航** - [AI_guide](https://github.com/Sayur1n/AI_guide) - 全面的AI工具导航指南
- **🖼️ 图像分割+翻译替换** - [sam2](https://github.com/Sayur1n/sam2) - 基于SAM的图像分割与翻译替换
- **🔍 AI图片鉴别** - [Community-Forensics](https://github.com/Sayur1n/Community-Forensics) - AI生成图片检测工具
- **📈 AI股票分析** - [TradingAgents-CN](https://github.com/Sayur1n/TradingAgents-CN) - 智能股票交易代理系统
- **👗 AI试衣** - [Ai_tryon](https://github.com/Sayur1n/Ai_tryon) - AI虚拟试衣技术
- **🌐 AI流式翻译** - [faster-whisper](https://github.com/Sayur1n/faster-whisper) - 高性能语音转录与实时翻译系统

## 项目特色

本项目不仅包含高性能的 Whisper 语音转录功能，还集成了基于有道翻译API的实时音频翻译系统，为用户提供完整的语音处理解决方案。

### 🚀 核心功能
- **高性能语音转录**: 基于 CTranslate2 的 Whisper 模型，速度提升4倍
- **实时音频翻译**: 支持麦克风和系统音频采集的实时翻译
- **多语言支持**: 支持中文、英文、日语、韩语、法语、德语、西班牙语、俄语等
- **低延迟处理**: 优化实时性能，适合会议、直播等场景

## 性能基准

### Whisper 模型性能对比

以下是转录 [**13分钟音频**](https://www.youtube.com/watch?v=0u7tTptBo9I) 所需的时间和内存使用情况对比：

* [openai/whisper](https://github.com/openai/whisper)@[v20240930](https://github.com/openai/whisper/tree/v20240930)
* [whisper.cpp](https://github.com/ggerganov/whisper.cpp)@[v1.7.2](https://github.com/ggerganov/whisper.cpp/tree/v1.7.2)
* [transformers](https://github.com/huggingface/transformers)@[v4.46.3](https://github.com/huggingface/transformers/tree/v4.46.3)
* [faster-whisper](https://github.com/SYSTRAN/faster-whisper)@[v1.1.0](https://github.com/SYSTRAN/faster-whisper/tree/v1.1.0)

### GPU 上的 Large-v2 模型性能

| 实现方式 | 精度 | Beam大小 | 时间 | VRAM使用 |
| --- | --- | --- | --- | --- |
| openai/whisper | fp16 | 5 | 2m23s | 4708MB |
| whisper.cpp (Flash Attention) | fp16 | 5 | 1m05s | 4127MB |
| transformers (SDPA)[^1] | fp16 | 5 | 1m52s | 4960MB |
| faster-whisper | fp16 | 5 | 1m03s | 4525MB |
| faster-whisper (`batch_size=8`) | fp16 | 5 | 17s | 6090MB |
| faster-whisper | int8 | 5 | 59s | 2926MB |
| faster-whisper (`batch_size=8`) | int8 | 5 | 16s | 4500MB |

### GPU 上的 distil-whisper-large-v3 模型性能

| 实现方式 | 精度 | Beam大小 | 时间 | YT Commons WER |
| --- | --- | --- | --- | --- |
| transformers (SDPA) (`batch_size=16`) | fp16 | 5 | 46m12s | 14.801 |
| faster-whisper (`batch_size=16`) | fp16 | 5 | 25m50s | 13.527 |

*GPU 基准测试在 NVIDIA RTX 3070 Ti 8GB 上使用 CUDA 12.4 执行。*
[^1]: transformers 在 batch_size > 1 时会出现内存不足

### CPU 上的 Small 模型性能

| 实现方式 | 精度 | Beam大小 | 时间 | 内存使用 |
| --- | --- | --- | --- | --- |
| openai/whisper | fp32 | 5 | 6m58s | 2335MB |
| whisper.cpp | fp32 | 5 | 2m05s | 1049MB |
| whisper.cpp (OpenVINO) | fp32 | 5 | 1m45s | 1642MB |
| faster-whisper | fp32 | 5 | 2m37s | 2257MB |
| faster-whisper (`batch_size=8`) | fp32 | 5 | 1m06s | 4230MB |
| faster-whisper | int8 | 5 | 1m42s | 1477MB |
| faster-whisper (`batch_size=8`) | int8 | 5 | 51s | 3608MB |

*在 Intel Core i7-12700K 上使用8线程执行。*

## 系统要求

* Python 3.9 或更高版本

与 openai-whisper 不同，本系统**不需要**在系统上安装 FFmpeg。音频通过 Python 库 [PyAV](https://github.com/PyAV-Org/PyAV) 解码，该库在其包中捆绑了 FFmpeg 库。

### GPU 要求

GPU 执行需要安装以下 NVIDIA 库：

* [cuBLAS for CUDA 12](https://developer.nvidia.com/cublas)
* [cuDNN 9 for CUDA 12](https://developer.nvidia.com/cudnn)

**注意**: 最新版本的 `ctranslate2` 仅支持 CUDA 12 和 cuDNN 9。对于 CUDA 11 和 cuDNN 8，当前解决方案是降级到 `ctranslate2` 的 `3.24.0` 版本，对于 CUDA 12 和 cuDNN 8，降级到 `ctranslate2` 的 `4.4.0` 版本（可以通过 `pip install --force-reinstall ctranslate2==4.4.0` 或在 `requirements.txt` 中指定版本来完成）。

## 安装

### 基础安装

可以从 [PyPI](https://pypi.org/project/faster-whisper/) 安装模块：

```bash
pip install faster-whisper
```

### 实时翻译功能依赖

```bash
pip install numpy sounddevice soundfile soundcard websocket-client pythoncom python-dotenv
```

### 其他安装方法

<details>
<summary>其他安装方法（点击展开）</summary>

#### 安装主分支

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
```

#### 安装特定提交

```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/a4f1cc8f11433e454c3934442b5e1a4ed5e865c3.tar.gz"
```

</details>

## 使用方法

### 基础语音转录

```python
from faster_whisper import WhisperModel

model_size = "large-v3"

# 在 GPU 上使用 FP16 运行
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# 或在 GPU 上使用 INT8 运行
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# 或在 CPU 上使用 INT8 运行
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)

print("检测到语言 '%s'，概率为 %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

**警告**: `segments` 是一个*生成器*，只有在迭代时才开始转录。可以通过将片段收集到列表或使用 for 循环来运行转录完成：

```python
segments, _ = model.transcribe("audio.mp3")
segments = list(segments)  # 转录将在这里实际运行
```

### 批量转录

以下代码片段演示了如何在示例音频文件上运行批量转录。`BatchedInferencePipeline.transcribe` 是 `WhisperModel.transcribe` 的直接替代品：

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel("turbo", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)
segments, info = batched_model.transcribe("audio.mp3", batch_size=16)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

### 词级时间戳

```python
segments, _ = model.transcribe("audio.mp3", word_timestamps=True)

for segment in segments:
    for word in segment.words:
        print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
```

### VAD 过滤器

该库集成了 [Silero VAD](https://github.com/snakers4/silero-vad) 模型来过滤没有语音的音频部分：

```python
segments, _ = model.transcribe("audio.mp3", vad_filter=True)
```

默认行为是保守的，只移除超过2秒的静音。可以在[源代码](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py)中查看可用的 VAD 参数和默认值。它们可以通过字典参数 `vad_parameters` 进行自定义：

```python
segments, _ = model.transcribe(
    "audio.mp3",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
```

批量转录默认启用 VAD 过滤器。

## 实时音频翻译

本项目提供两种实时音频翻译解决方案，满足不同场景的需求：

### 方案一：本地转录 + API翻译 (`realtime_asr_trans_gui.py`)

**特点**: 本地Whisper模型进行语音识别 + 有道翻译API进行文本翻译

**优势**:
- 🎯 **高精度**: 使用本地faster-whisper模型，转录准确率高
- 🔒 **隐私保护**: 语音识别在本地完成，音频数据不离开本地
- ⚡ **低延迟**: 本地ASR处理快速，适合实时应用
- 🎨 **智能标点**: 基于停顿时间的智能标点系统
- 🔄 **增量翻译**: 支持实时增量式翻译，体验流畅

**适用场景**:
- 对转录精度要求高的会议记录
- 需要保护隐私的敏感场景
- 本地化部署需求
- 离线环境下的语音处理

**使用方法**:
```bash
# 基础使用（麦克风输入，日语→中文）
python realtime_asr_trans_gui.py --trans-from ja --trans-to zh-CHS

# 系统音频回环采集
python realtime_asr_trans_gui.py --source system --trans-from en --trans-to zh-CHS

# 自定义模型和参数
python realtime_asr_trans_gui.py \
    --model large-v3 \
    --device cuda \
    --compute-type float16 \
    --trans-from ja \
    --trans-to zh-CHS \
    --chunk-sec 0.6 \
    --accum-sec 1.8
```

**主要参数**:
- `--model`: Whisper模型大小 (tiny/base/small/medium/large-v2/large-v3/distil-large-v3)
- `--device`: 推理设备 (cuda/cpu)
- `--compute-type`: 计算精度 (float16/int8)
- `--trans-from/--trans-to`: 翻译语言对
- `--chunk-sec`: 音频块大小(秒)
- `--accum-sec`: 累积时间阈值(秒)

### 方案二：纯API语音翻译 (`realtime_translation_demo.py`)

**特点**: 完全基于有道流式语音翻译API，无需本地模型

**优势**:
- 🚀 **快速部署**: 无需下载模型，开箱即用
- 💰 **成本低**: 按API调用量计费，适合轻量使用
- 🌐 **云端优化**: 有道专业语音识别和翻译服务
- 📱 **轻量级**: 程序体积小，依赖少
- 🔄 **流式处理**: 真正的流式语音翻译，延迟更低

**适用场景**:
- 快速原型开发和测试
- 轻量级应用部署
- 云端优先的解决方案
- 对部署便捷性要求高的场景

**使用方法**:
```bash
# 基础使用（需要提供API密钥）
python realtime_translation_demo.py \
    --yd-app-key YOUR_APP_KEY \
    --yd-app-secret YOUR_APP_SECRET \
    --trans-from ja \
    --trans-to zh-CHS

# 系统音频回环
python realtime_translation_demo.py \
    --source system \
    --yd-app-key YOUR_APP_KEY \
    --yd-app-secret YOUR_APP_SECRET \
    --trans-from en \
    --trans-to zh-CHS

# 调整音频参数
python realtime_translation_demo.py \
    --yd-app-key YOUR_APP_KEY \
    --yd-app-secret YOUR_APP_SECRET \
    --chunk-sec 0.4 \
    --vol-gain 1.2
```

**主要参数**:
- `--yd-app-key/--yd-app-secret`: 有道API密钥（必需）
- `--trans-from/--trans-to`: 翻译语言对
- `--source`: 音频来源 (mic/system)
- `--chunk-sec`: 音频块大小(秒)
- `--vol-gain`: 音量增益

### 两种方案对比

| 特性 | 本地转录+API翻译 | 纯API语音翻译 |
|------|------------------|---------------|
| **转录精度** | ⭐⭐⭐⭐⭐ (本地Whisper) | ⭐⭐⭐⭐ (有道ASR) |
| **隐私保护** | ⭐⭐⭐⭐⭐ (本地处理) | ⭐⭐⭐ (云端处理) |
| **部署便捷性** | ⭐⭐⭐ (需下载模型) | ⭐⭐⭐⭐⭐ (开箱即用) |
| **成本** | ⭐⭐⭐⭐ (仅翻译API费用) | ⭐⭐⭐ (完整API费用) |
| **延迟** | ⭐⭐⭐⭐ (本地ASR) | ⭐⭐⭐⭐⭐ (流式处理) |
| **离线能力** | ⭐⭐⭐⭐⭐ (可完全离线) | ⭐⭐ (依赖网络) |

### 配置要求

1. **API 设置**: 在 [有道智云](https://ai.youdao.com/) 注册获取 `APP_KEY` 和 `APP_SECRET`
2. **环境变量**: 在根目录创建 `.env` 文件：
   ```bash
   APP_KEY=your_app_key_here
   APP_SECRET=your_app_secret_here
   AUDIO_PATH=path/to/your/audio/file.wav
   ```
3. **依赖安装**: 
   ```bash
   # 基础依赖
   pip install numpy sounddevice soundfile soundcard websocket-client pythoncom python-dotenv
   
   # 本地转录额外依赖
   pip install faster-whisper
   ```

### 支持的语言代码

有道翻译API支持的语言代码示例：
- `zh-CHS`: 简体中文
- `en`: 英语
- `ja`: 日语
- `ko`: 韩语
- `fr`: 法语
- `de`: 德语
- `es`: 西班牙语
- `ru`: 俄语

## 模型转换

当从大小加载模型时，如 `WhisperModel("large-v3")`，相应的 CTranslate2 模型会自动从 [Hugging Face Hub](https://huggingface.co/Systran) 下载。

我们还提供了一个脚本来转换与 Transformers 库兼容的任何 Whisper 模型。它们可以是原始的 OpenAI 模型或用户微调的模型。

例如，以下命令转换[原始的 "large-v3" Whisper 模型](https://huggingface.co/openai/whisper-large-v3)并以 FP16 保存权重：

```bash
pip install transformers[torch]>=4.23

ct2-transformers-converter --model openai/whisper-large-v3 --output_dir whisper-large-v3-ct2
--copy_files tokenizer.json preprocessor_config.json --quantization float16
```

* `--model` 选项接受 Hub 上的模型名称或模型目录的路径。
* 如果不使用 `--copy_files tokenizer.json` 选项，当稍后加载模型时会自动下载分词器配置。

也可以从代码中转换模型。请参阅[转换 API](https://opennmt.net/CTranslate2/python/ctranslate2.converters.TransformersConverter.html)。

### 加载转换后的模型

1. 直接从本地目录加载模型：
```python
model = faster_whisper.WhisperModel("whisper-large-v3-ct2")
```

2. [将您的模型上传到 Hugging Face Hub](https://huggingface.co/docs/transformers/model_sharing#upload-with-the-web-interface)并从其名称加载：
```python
model = faster_whisper.WhisperModel("username/whisper-large-v3-ct2")
```

## 社区集成

以下是使用 faster-whisper 的开源项目的不完整列表。欢迎将您的项目添加到列表中！

* [speaches](https://github.com/speaches-ai/speaches) 是一个使用 `faster-whisper` 的 OpenAI 兼容服务器。它易于使用 Docker 部署，与 OpenAI SDKs/CLI 配合使用，支持流式传输和实时转录。
* [WhisperX](https://github.com/m-bain/whisperX) 是一个获奖的 Python 库，使用 wav2vec2 对齐提供说话人分离和准确的词级时间戳
* [whisper-ctranslate2](https://github.com/Softcatala/whisper-ctranslate2) 是一个基于 faster-whisper 的命令行客户端，与 openai/whisper 的原始客户端兼容
* [whisper-diarize](https://github.com/MahmoudAshraf97/whisper-diarization) 是一个基于 faster-whisper 和 NVIDIA NeMo 的说话人分离工具
* [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) 适用于 Windows、Linux 和 macOS 的 faster-whisper 独立 CLI 可执行文件
* [asr-sd-pipeline](https://github.com/hedrergudene/asr-sd-pipeline) 提供了一个可扩展、模块化的端到端多说话人语音转文字解决方案，使用 AzureML 管道实现
* [Open-Lyrics](https://github.com/zh-plus/Open-Lyrics) 是一个 Python 库，使用 faster-whisper 转录语音文件，并使用 OpenAI-GPT 将结果文本翻译/润色为所需语言的 `.lrc` 文件
* [wscribe](https://github.com/geekodour/wscribe) 是一个灵活的转录生成工具，支持 faster-whisper，可以导出词级转录，然后可以使用 [wscribe-editor](https://github.com/geekodour/wscribe-editor) 编辑导出的转录
* [aTrain](https://github.com/BANDAS-Center/aTrain) 是 faster-whisper 的图形用户界面实现，由格拉茨大学 BANDAS-Center 开发，用于 Windows（[Windows Store App](https://apps.microsoft.com/detail/atrain/9N15Q44SZNS2)）和 Linux 的转录和说话人分离
* [Whisper-Streaming](https://github.com/ufal/whisper_streaming) 为离线 Whisper 类语音转文字模型实现实时模式，推荐使用 faster-whisper 作为后端。它实现了基于实际源复杂性的自适应延迟流式策略，并展示了最先进的技术
* [WhisperLive](https://github.com/collabora/WhisperLive) 是 OpenAI Whisper 的近实时实现，使用 faster-whisper 作为后端实时转录音频
* [Faster-Whisper-Transcriber](https://github.com/BBC-Esq/ctranslate2-faster-whisper-transcriber) 是一个简单但可靠的语音转录器，提供用户友好的界面
* [Open-dubbing](https://github.com/softcatala/open-dubbing) 是一个开放配音系统，使用机器学习模型自动翻译和同步不同语言的音频对话
* [Whisper-FastAPI](https://github.com/heimoshuiyu/whisper-fastapi) whisper-fastapi 是一个非常简单的脚本，提供与 OpenAI、HomeAssistant 和 Konele（Android 语音输入）格式兼容的 API 后端

## 性能对比注意事项

如果您正在与其他 Whisper 实现进行性能比较，请确保在相似设置下运行比较。特别是：

* 验证使用相同的转录选项，特别是相同的 beam 大小。例如，在 openai/whisper 中，`model.transcribe` 使用默认 beam 大小为 1，但这里我们使用默认 beam 大小为 5。
* 转录速度与转录中的单词数量密切相关，因此确保其他实现具有与此相似的 WER（词错误率）。
* 在 CPU 上运行时，确保设置相同的线程数。许多框架会读取环境变量 `OMP_NUM_THREADS`，可以在运行脚本时设置：

```bash
OMP_NUM_THREADS=4 python3 my_script.py
```

## 故障排除

### 常见问题

1. **找不到音频设备**
   ```bash
   python realtime_translation_demo.py --list-devices
   ```

2. **连接API失败**
   - 检查网络连接
   - 验证API密钥是否正确
   - 确认API服务状态

3. **音频质量差**
   - 调整 `--vol-gain` 参数
   - 检查麦克风设置
   - 确保环境安静

4. **翻译延迟高**
   - 减少 `--accum-sec` 值
   - 减少 `--chunk-sec` 值
   - 检查网络延迟

## 技术架构

```
音频采集 → 音频处理 → 有道翻译API → 结果处理 → 显示输出
    ↓           ↓           ↓           ↓         ↓
 麦克风/    格式转换     WebSocket     JSON解析   实时显示
系统音频    16kHz PCM    流式传输     结果提取   状态监控
```

## 开发说明

程序基于以下技术栈：
- **音频处理**: sounddevice, soundcard, numpy
- **网络通信**: websocket-client
- **API集成**: 有道翻译WebSocket API
- **异步处理**: threading, queue

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！
