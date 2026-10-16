# ComfyUI 音频分割插件 (ComfyUI-AudioSegment)

这是一个为 ComfyUI 设计的自定义节点套件，用于高级音频处理。它允许用户通过两种方式将一个长的音频文件分割成多个片段：
1.  **基于静音**：根据静音位置和设定的最大时长，智能地分割成多个较短的音频片段列表。
2.  **基于文本段落**：根据用户提供的参考文本，利用 AI 语音识别进行精准对齐和分割。

## 功能

此插件套件包含五个协同工作的节点：

### 1. Audio Segment By Paragraph (按段落分割音频)
这是新增的核心功能节点。它能根据用户提供的带段落标记的文本，利用 Whisper ASR 模型进行语音到文本的对齐，从而实现精准的按段落分割。

-   **输入**:
    -   `audio`: 原始的 ComfyUI 音频对象。
    -   `text`: 多行字符串，每个段落占一行。程序会根据换行符切分段落。
    -   `model`: 一个下拉菜单，用于选择要使用的 Whisper 模型。支持所有标准模型（如 `base`, `large-v3`）、本地的 `.pt` 文件，以及存放在 `ComfyUI/models/stt/whisper/` 目录下的 Hugging Face 模型文件夹。如果本地找不到所选模型，会自动从网络下载。
-   **输出**:
    -   `AudioClip list`: 一个自定义的 `AudioClip` 对象列表。每个 `AudioClip` 包含 `audio` (ComfyUI 音频对象), `startTime` (毫秒), `endTime` (毫秒), 和 `text` (对应的原始文本段落)。
    -   `size`: `AudioClip` 列表的长度。
    -   `srt`: 根据分割结果生成的 SRT 格式字幕字符串。

### 2. Select AudioClip From List (从AudioClip列表中选择)
这个节点允许你从 `Audio Segment By Paragraph` 生成的列表中选择一个特定的 `AudioClip` 进行后续处理。

-   **输入**:
    -   `audio_clip_list`: 从 `Audio Segment By Paragraph` 输出的 `AudioClip` 列表。
    -   `index`: 要选择的 `AudioClip` 在列表中的索引（从 0 开始）。
-   **输出**:
    -   `audio`: `AudioClip` 中包含的 ComfyUI 音频对象。
    -   `text`: `AudioClip` 中包含的文本字符串。

### 3. Audio Segmenter (音频分割器 - 基于静音)
这是原始的核心节点，负责根据静音将音频分割成一个片段列表。

-   **输入**:
    -   `audio`: 原始的 ComfyUI 音频对象。
    -   `max_length_s`: 每个输出片段的最大长度（单位：秒）。
    -   `silence_thresh_db`: 静音阈值（单位：分贝）。低于此音量的部分被视作静音。
-   **输出**:
    -   `audio_list`: 一个包含多个独立音频片段的列表。
    -   `count`: 列表中音频片段的数量。

### 4. Select Audio From List (从列表中选择音频)
这个节点允许你从 `Audio Segmenter` 生成的列表中选择一个特定的音频片段进行后续处理。

-   **输入**:
    -   `audio_list`: 从 `Audio Segmenter` 输出的音频列表。
    -   `index`: 要选择的音频片段在列表中的索引（从 0 开始）。
-   **输出**:
    -   `AUDIO`: 列表中指定索引位置的单个音频片段。

### 5. Audio List to Batch (列表转批处理)
这个节点将一个音频列表转换回 ComfyUI 标准的批处理（batch）音频格式，方便与其他需要批处理输入的节点（如 ASR 模型）兼容。

-   **输入**:
    -   `audio_list`: 从 `Audio Segmenter` 输出的音频列表。
-   **输出**:
    -   `AUDIO`: 一个包含列表中所有音频片段的批处理音频对象。所有片段会被自动填充（padding）到与最长片段相同的长度。

## 安装方法

1.  **导航到 `custom_nodes` 目录**:
    打开终端，进入你的 ComfyUI 安装目录下的 `custom_nodes` 文件夹。
    ```bash
    cd path/to/ComfyUI/custom_nodes/
    ```

2.  **克隆仓库**:
    将本仓库克隆到 `custom_nodes` 目录下。
    ```bash
    git clone git@gitee.com:dreamidea/comfyui-audio-segment.git
    ```

3.  **安装依赖**:
    进入插件目录，然后使用 ComfyUI 环境下的 `pip` 安装所需的依赖库。
    ```bash
    cd comfyui-audio-segment
    pip install -r requirements.txt
    ```
    *(注意: 如果你的 ComfyUI 是通过虚拟环境（如 venv 或 conda）运行的，请确保先激活该环境。)*

4.  **重启 ComfyUI**:
    完全关闭并重新启动 ComfyUI。之后你就可以在 "Audio/Segmentation" 类别下找到所有新节点了。

## 实现原理

### 基于静音的分割
插件的原始分割逻辑位于 `audio_segmenter.py` 文件中，主要依赖 `pydub` 库实现。它通过检测音频中的静音部分来执行初步分割，然后通过智能合并与强制切分，使每个片段的长度尽可能接近用户设定的最大值。

### 基于文本段落的分割
新增的核心逻辑位于 `audio_paragraph_segmenter.py` 文件中，其工作流程如下：

1.  **语音识别 (ASR)**:
    -   利用 `openai-whisper` 库对输入的音频进行高精度语音识别，并获取每个词语的精确时间戳。
    -   支持动态加载多种 Whisper 模型，包括标准模型（如 `base`, `large-v3`）、本地 `.pt` 文件以及存放在 `ComfyUI/models/stt/whisper/` 目录下的 Hugging Face 模型。如果模型文件不存在，程序会自动从网络下载。

2.  **文本对齐**:
    -   由于用户的参考文本与 Whisper 的识别结果可能存在差异（如同音字、繁简体、漏字等），插件实现了一套健壮的文本对齐算法。
    -   该算法首先使用 `opencc-python-reimplemented` 库将参考文本和识别结果统一转换为简体中文，以消除繁简体差异。
    -   然后，利用 Python 的 `difflib.SequenceMatcher` 找出两个文本之间所有的匹配块，通过计算累积相似度来容忍 ASR 的零散错误，从而为每个段落精确地定位其在音频中的起止时间。

3.  **音频切分与数据封装**:
    -   根据对齐算法计算出的时间戳，使用 `pydub` 将原始音频精确切分成多个片段。
    -   每个切分后的音频片段，连同其对应的原始文本段落和起止时间，被封装到一个自定义的 `AudioClip` 对象中，方便后续节点（如 `Select AudioClip From List`）进行调用。
