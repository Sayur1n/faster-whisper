# 实时音频翻译程序

这是一个基于有道翻译API的实时音频翻译程序，支持麦克风和系统音频采集，能够实时将语音转换为文字并进行翻译。

## 功能特点

- 🎤 支持麦克风输入和系统音频回环采集
- 🌐 集成有道翻译API，支持多种语言翻译
- ⚡ 实时音频处理，低延迟翻译
- 🔧 可配置的音频参数和翻译设置
- 📊 实时状态监控和结果统计

## 系统要求

- Python 3.7+
- Windows 10/11 (支持系统音频回环)
- 有道翻译API账号和应用密钥

## 安装依赖

```bash
pip install numpy sounddevice soundfile soundcard websocket-client pythoncom
```

## 配置有道翻译API

1. 访问 [有道智云](https://ai.youdao.com/) 注册账号
2. 创建应用获取 `APP_KEY` 和 `APP_SECRET`
3. 在程序中替换以下配置：

```python
APP_KEY = 'your_app_key_here'
APP_SECRET = 'your_app_secret_here'
```

## 使用方法

### 基本用法

```bash
# 使用麦克风输入，中文翻译为英文
python realtime_translation_demo.py

# 使用系统音频回环，中文翻译为英文
python realtime_translation_demo.py --source system

# 指定源语言和目标语言
python realtime_translation_demo.py --lang-from ja --lang-to en

# 调整音频参数
python realtime_translation_demo.py --chunk-sec 0.5 --accum-sec 1.5
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source` | 音频来源：`mic`(麦克风) 或 `system`(系统音频) | `mic` |
| `--lang-from` | 源语言代码 | `zh-CHS` |
| `--lang-to` | 目标语言代码 | `en` |
| `--chunk-sec` | 单次音频块大小(秒) | `0.6` |
| `--accum-sec` | 累积时间阈值(秒) | `2.0` |
| `--vol-gain` | 音量增益 | `1.0` |
| `--app-key` | 有道API应用ID | 程序内置值 |
| `--app-secret` | 有道API应用密钥 | 程序内置值 |
| `--list-devices` | 列出可用音频设备 | - |

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

## 使用场景

### 1. 实时会议翻译
- 使用系统音频回环采集会议音频
- 实时翻译为多种语言
- 适合国际会议和远程协作

### 2. 语音学习辅助
- 使用麦克风输入练习发音
- 实时翻译帮助理解
- 支持多种语言学习

### 3. 直播翻译
- 采集直播音频流
- 实时字幕翻译
- 提升国际观众体验

## 注意事项

1. **音频格式要求**: 有道API要求16kHz采样率、单声道、16位PCM格式
2. **网络连接**: 需要稳定的网络连接访问有道翻译API
3. **API限制**: 注意有道API的调用频率和配额限制
4. **隐私安全**: 音频数据会发送到有道服务器，注意隐私保护

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
