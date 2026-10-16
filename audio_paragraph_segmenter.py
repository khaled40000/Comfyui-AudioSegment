import whisper
import tempfile
import os
import difflib
import re
from pydub import AudioSegment
from typing import List, Dict, Any
import opencc

# 尝试导入 transformers，如果失败则优雅降级
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not found. Falling back to openai-whisper only.")

def transcribe_with_word_timestamps(audio_path: str, model_name: str, model_dir: str) -> List[Dict[str, Any]]:
    """
    使用 Whisper 对音频文件进行转录，并返回包含词级别时间戳的扁平化词语列表。
    - 如果 model_name 是一个目录名，则使用 HuggingFace transformers。
    - 否则，使用 openai-whisper（处理 .pt 文件和标准模型）。
    """
    model_path = os.path.join(model_dir, model_name)
    all_words = []

    try:
        # 检查是否为 HuggingFace 模型（文件夹形式）
        if TRANSFORMERS_AVAILABLE and os.path.isdir(model_path):
            print(f"Loading HuggingFace model '{model_name}' from directory: {model_path}")
            
            # 使用 transformers pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_path,
                torch_dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            
            result = pipe(audio_path, return_timestamps="word", chunk_length_s=30)
            
            # 将 transformers 的输出格式转换为 openai-whisper 的格式
            if 'chunks' in result:
                for chunk in result['chunks']:
                    all_words.append({
                        'word': chunk['text'].strip(),
                        'start': chunk['timestamp'][0],
                        'end': chunk['timestamp'][1],
                    })
        else:
            # 使用 openai-whisper
            print(f"Loading openai-whisper model: '{model_name}'")
            model = whisper.load_model(model_name, download_root=model_dir)
            result = model.transcribe(audio_path, word_timestamps=True, verbose=False)
            
            # 将结果扁平化为单个词语列表
            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    all_words.append(word)
        
        return all_words

    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return []

def align_paragraphs_to_words(paragraphs: List[str], whisper_words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    使用 difflib 将输入的段落列表与 Whisper 的词语转录结果对齐。
    返回一个包含每个段落的 'text', 'start', 和 'end' 时间的列表。
    """
    if not whisper_words or not paragraphs:
        return []
    
    converter = opencc.OpenCC('t2s')

    def normalize_text(text):
        """A simple normalization for Chinese: convert to simplified, remove punctuation, spaces, and convert to lower case."""
        simplified_text = converter.convert(str(text))
        return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', simplified_text).lower()

    whisper_full_text = "".join(normalize_text(w['word']) for w in whisper_words)
    
    aligned_segments = []
    current_search_start_char = 0

    for para in paragraphs:
        normalized_para = normalize_text(para)
        if not normalized_para:
            continue

        search_space = whisper_full_text[current_search_start_char:]
        matcher = difflib.SequenceMatcher(None, search_space, normalized_para, autojunk=False)
        
        matching_blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]

        if not matching_blocks:
            continue

        total_match_size = sum(b.size for b in matching_blocks)
        ratio = total_match_size / len(normalized_para) if len(normalized_para) > 0 else 0

        if ratio < 0.5:  # Similarity threshold
            continue

        first_block = matching_blocks[0]
        last_block = matching_blocks[-1]

        match_start_char_relative = first_block.a
        match_end_char_relative = last_block.a + last_block.size

        match_start_char_abs = current_search_start_char + match_start_char_relative
        match_end_char_abs = current_search_start_char + match_end_char_relative

        start_word_idx, end_word_idx = -1, -1
        char_cursor = 0
        for i, word_info in enumerate(whisper_words):
            word_text = normalize_text(word_info['word'])
            word_len = len(word_text)
            
            if start_word_idx == -1 and char_cursor + word_len > match_start_char_abs:
                start_word_idx = i
            
            if start_word_idx != -1 and char_cursor + word_len >= match_end_char_abs:
                end_word_idx = i
                break
            
            char_cursor += word_len

        if start_word_idx == -1 or end_word_idx == -1:
            continue
            
        start_time = whisper_words[start_word_idx]['start']
        end_time = whisper_words[end_word_idx]['end']

        aligned_segments.append({
            "text": para.strip(),
            "start": start_time,
            "end": end_time
        })

        char_cursor = 0
        for i in range(end_word_idx + 1):
             char_cursor += len(normalize_text(whisper_words[i]['word']))
        current_search_start_char = char_cursor
        
        if current_search_start_char >= len(whisper_full_text):
            break

    return aligned_segments

def format_time(seconds: float) -> str:
    """
    将秒数格式化为 SRT 时间戳格式 (HH:MM:SS,ms)。
    """
    millis = int(seconds * 1000)
    ss, ms = divmod(millis, 1000)
    mm, ss = divmod(ss, 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def generate_srt(aligned_segments: List[Dict[str, Any]]) -> str:
    """
    根据对齐后的片段生成 SRT 格式的字幕。
    """
    srt_content = []
    for i, segment in enumerate(aligned_segments, 1):
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        text = segment['text']
        srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
    return "\n".join(srt_content)

def segment_by_paragraph(audio_data: bytes, sample_rate: int, channels: int, text: str, model_name: str, model_dir: str):
    """
    主函数，协调整个流程：转录 -> 对齐 -> 切分。
    """
    # 1. 将内存中的音频数据保存到临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        audio_segment = AudioSegment(
            data=audio_data,
            sample_width=2, # 16-bit
            frame_rate=sample_rate,
            channels=channels
        )
        audio_segment.export(temp_audio_file.name, format="wav")
        temp_audio_path = temp_audio_file.name

    try:
        # 2. 调用 Whisper 获取带时间戳的词语
        whisper_result = transcribe_with_word_timestamps(temp_audio_path, model_name, model_dir)

        # 3. 清理和准备输入文本
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if not paragraphs:
            return [], 0, ""

        # 4. 对齐文本和音频
        aligned_segments = align_paragraphs_to_words(paragraphs, whisper_result)

        # 5. 根据对齐结果切分音频并创建 AudioClip
        audio_clips = []
        original_audio = AudioSegment.from_wav(temp_audio_path)

        for segment_info in aligned_segments:
            start_ms = int(segment_info['start'] * 1000)
            end_ms = int(segment_info['end'] * 1000)
            
            # 使用 pydub 切分
            audio_chunk = original_audio[start_ms:end_ms]

            audio_clips.append({
                "audio": audio_chunk,
                "startTime": start_ms,
                "endTime": end_ms,
                "text": segment_info['text']
            })

        # 6. 生成 SRT 字符串
        srt_string = generate_srt(aligned_segments)
        
        return audio_clips, len(audio_clips), srt_string

    finally:
        # 7. 清理临时文件
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
    
    return [], 0, ""
