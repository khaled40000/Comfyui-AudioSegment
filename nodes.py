import torch
import numpy as np
import folder_paths
import os
from .audio_segmenter import segment_audio
from .audio_paragraph_segmenter import segment_by_paragraph

# Whisper模型路径
stt_model_dir = os.path.join(folder_paths.models_dir, "stt", "whisper")
os.makedirs(stt_model_dir, exist_ok=True)

# 获取可用模型列表
local_models = []
if os.path.isdir(stt_model_dir):
    for item in os.listdir(stt_model_dir):
        # 如果是文件夹，直接添加
        if os.path.isdir(os.path.join(stt_model_dir, item)):
            local_models.append(item)
        # 如果是 .pt 文件，去掉后缀添加
        elif item.endswith(".pt"):
            local_models.append(os.path.splitext(item)[0])

standard_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
available_models = sorted(list(set(local_models + standard_models)))


# 自定义类型，用于 ComfyUI 识别
class AudioClip:
    @classmethod
    def is_list(cls, item):
        return isinstance(item, list)

# 在 ComfyUI 中注册自定义类型
# 注意：这部分可能需要根据 ComfyUI 的插件系统进行调整
# 通常，我们通过返回一个特殊的字符串或 Python 类型来处理
# 在这里，我们先用"*"作为通配符，然后在节点中处理具体逻辑

class AudioSegmentByParagraphNode:
    AUX_ID = "dreamidea/comfyui-audio-segment"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "在此处输入按段落分隔的文本..."
                }),
                "model": (available_models, ),
            }
        }

    RETURN_TYPES = ("*", "INT", "STRING",)
    RETURN_NAMES = ("AudioClip list", "size", "srt",)
    FUNCTION = "segment"
    CATEGORY = "Audio/Segmentation"

    def segment(self, audio, text, model):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.device.type != 'cpu':
            waveform = waveform.to('cpu')

        # 假设 batch size 为 1
        waveform_single = waveform[0]
        waveform_np = waveform_single.numpy()
        waveform_int16 = (waveform_np * 32767).astype(np.int16)
        
        channels = waveform_int16.shape[0]
        audio_data = waveform_int16.T.tobytes()

        # 调用新的分割函数
        audio_clips_raw, count, srt_string = segment_by_paragraph(
            audio_data=audio_data,
            sample_rate=sample_rate,
            channels=channels,
            text=text,
            model_name=model,
            model_dir=stt_model_dir
        )

        if not audio_clips_raw:
            return ([], 0, "")

        # 将 pydub 音频块转换回 ComfyUI 的 AUDIO 格式
        final_audio_clips = []
        for clip in audio_clips_raw:
            pydub_chunk = clip["audio"]
            
            chunk_np = np.array(pydub_chunk.get_array_of_samples())
            
            if pydub_chunk.channels > 1:
                chunk_np = chunk_np.reshape((-1, pydub_chunk.channels)).T
            else:
                chunk_np = chunk_np.reshape((1, -1))

            chunk_tensor = torch.from_numpy(chunk_np.astype(np.float32) / 32767.0)
            
            comfy_audio = {
                "waveform": chunk_tensor.unsqueeze(0),
                "sample_rate": sample_rate
            }

            final_audio_clips.append({
                "audio": comfy_audio,
                "startTime": clip["startTime"],
                "endTime": clip["endTime"],
                "text": clip["text"]
            })
            
        return (final_audio_clips, count, srt_string)

class AudioSegmenterNode:
    AUX_ID = "dreamidea/comfyui-audio-segment"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "max_length_s": ("INT", {
                    "default": 10, "min": 1, "max": 3600, "step": 1
                }),
                "silence_thresh_db": ("INT", {
                    "default": -40, "min": -100, "max": 0, "step": 1
                }),
            }
        }

    RETURN_TYPES = ("*", "INT",)
    RETURN_NAMES = ("audio_list", "count",)
    FUNCTION = "segment"
    CATEGORY = "Audio/Segmentation"

    def segment(self, audio, max_length_s, silence_thresh_db):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.device.type != 'cpu':
            waveform = waveform.to('cpu')

        waveform_single = waveform[0]
        waveform_np = waveform_single.numpy()
        waveform_int16 = (waveform_np * 32767).astype(np.int16)
        
        channels = waveform_int16.shape[0]
        # Transpose to interleave channels for pydub
        audio_data = waveform_int16.T.tobytes()

        chunks = segment_audio(
            audio_data=audio_data,
            frame_rate=sample_rate,
            channels=channels,
            max_length_s=max_length_s,
            silence_thresh_db=silence_thresh_db
        )

        if not chunks:
            return ([], 0)

        audio_list = []
        for chunk in chunks:
            chunk_np = np.array(chunk.get_array_of_samples())
            
            if chunk.channels > 1:
                chunk_np = chunk_np.reshape((-1, chunk.channels)).T
            else:
                chunk_np = chunk_np.reshape((1, -1))

            chunk_tensor = torch.from_numpy(chunk_np.astype(np.float32) / 32767.0)
            
            # 每个音频片段都是一个独立的 AUDIO 对象 (batch size 1)
            audio_list.append({
                "waveform": chunk_tensor.unsqueeze(0),
                "sample_rate": sample_rate
            })
        
        return (audio_list, len(audio_list))

class SelectAudioClipFromList:
    AUX_ID = "dreamidea/comfyui-audio-segment"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_clip_list": ("*",),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "text",)
    FUNCTION = "select_clip"
    CATEGORY = "Audio/Segmentation"

    def select_clip(self, audio_clip_list, index):
        if not audio_clip_list or index >= len(audio_clip_list):
            # Fallback to empty audio and text if list is empty or index is out of bounds
            empty_waveform = torch.zeros((1, 1, 1))
            empty_audio = {"waveform": empty_waveform, "sample_rate": 44100}
            return (empty_audio, "")
        
        selected_clip = audio_clip_list[index]
        return (selected_clip["audio"], selected_clip["text"])

class SelectAudioFromList:
    AUX_ID = "dreamidea/comfyui-audio-segment"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_list": ("*",),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "select_audio"
    CATEGORY = "Audio/Segmentation"

    def select_audio(self, audio_list, index):
        if not audio_list or index >= len(audio_list):
            # 如果列表为空或索引越界，返回一个空的音频对象
            sample_rate = 44100  # Fallback sample rate
            if audio_list:
                sample_rate = audio_list[0].get("sample_rate", sample_rate)
            empty_waveform = torch.zeros((1, 1, 1))
            return ({"waveform": empty_waveform, "sample_rate": sample_rate},)
        
        return (audio_list[index],)

class AudioListToBatch:
    AUX_ID = "dreamidea/comfyui-audio-segment"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_list": ("*",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "convert_to_batch"
    CATEGORY = "Audio/Segmentation"

    def convert_to_batch(self, audio_list):
        if not audio_list:
            empty_waveform = torch.zeros((1, 1, 1))
            return ({"waveform": empty_waveform, "sample_rate": 44100},)

        waveforms = [item["waveform"] for item in audio_list]
        sample_rate = audio_list[0]["sample_rate"]

        # 确保所有 waveform 的 batch size 都是 1
        waveforms = [w[0] for w in waveforms]

        max_samples = max(t.shape[1] for t in waveforms)
        
        padded_tensors = []
        for t in waveforms:
            padding_needed = max_samples - t.shape[1]
            if padding_needed > 0:
                padded_tensor = torch.nn.functional.pad(t, (0, padding_needed))
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(t)

        batched_waveform = torch.stack(padded_tensors, dim=0)

        return ({"waveform": batched_waveform, "sample_rate": sample_rate},)
