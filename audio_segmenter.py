import io
from pydub import AudioSegment
from pydub.silence import split_on_silence

def segment_audio(audio_data: bytes, frame_rate: int, channels: int, max_length_s: int, silence_thresh_db: int, min_silence_len_ms: int = 500, keep_silence_ms: int = 250):
    """
    将音频根据静音和最大长度进行分割。
    新逻辑：优先在静音处分割，然后将短片段合并，使其尽可能接近但不超过最大长度。

    :param audio_data: 原始音频数据 (bytes).
    :param frame_rate: 音频的采样率.
    :param channels: 音频的通道数.
    :param max_length_s: 每个片段的最大长度（秒）.
    :param silence_thresh_db: 低于此分贝值被认为是静音.
    :param min_silence_len_ms: 最小静音长度（毫秒），用于分割.
    :param keep_silence_ms: 在分割后的片段开头和结尾保留的静音长度（毫秒）.
    :return: 一个包含 pydub.AudioSegment 对象的列表.
    """
    if not audio_data:
        raise ValueError("输入的音频数据不能为空。")

    # 从原始字节创建 pydub AudioSegment
    # sample_width=2 对应 16-bit PCM
    audio = AudioSegment(
        data=audio_data,
        sample_width=2,
        frame_rate=frame_rate,
        channels=channels
    )

    max_length_ms = max_length_s * 1000

    # 1. 首先根据静音进行分割
    initial_chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db,
        # keep_silence=keep_silence_ms
        keep_silence=True
    )

    if not initial_chunks:
        initial_chunks = [audio]

    # 2. 合并短片段，使其长度接近 max_length_ms
    combined_chunks = []
    current_chunk = None

    for chunk in initial_chunks:
        if current_chunk is None:
            current_chunk = chunk
        elif len(current_chunk) + len(chunk) <= max_length_ms:
            current_chunk += chunk
        else:
            combined_chunks.append(current_chunk)
            current_chunk = chunk
    
    if current_chunk is not None:
        combined_chunks.append(current_chunk)

    # 3. 再次检查是否有片段超过最大长度（如果单个静音分割后的片段就超长了）
    #    并将其暴力分割
    final_chunks = []
    for chunk in combined_chunks:
        if len(chunk) > max_length_ms:
            for i in range(0, len(chunk), max_length_ms):
                final_chunks.append(chunk[i:i + max_length_ms])
        else:
            final_chunks.append(chunk)

    return final_chunks
