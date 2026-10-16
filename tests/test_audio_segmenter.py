import unittest
import numpy as np
from pydub import AudioSegment
import sys
import os

# 将父目录添加到 sys.path 以便导入 audio_segmenter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audio_segmenter import segment_audio

class TestAudioSegmenter(unittest.TestCase):

    def generate_test_audio(self, sample_rate=44100, channels=1, duration_s=30):
        """
        生成一段测试音频：
        - 0-5s: 声音
        - 5-10s: 静音
        - 10-15s: 声音
        - 15-20s: 静音
        - 20-30s: 声音 (长片段)
        """
        samples = int(duration_s * sample_rate)
        t = np.linspace(0., duration_s, samples)
        
        # 生成 440 Hz 正弦波
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * 440. * t)
        
        # 插入静音
        data[int(5 * sample_rate):int(10 * sample_rate)] = 0
        data[int(15 * sample_rate):int(20 * sample_rate)] = 0
        
        # 转换为 int16
        audio_data_np = data.astype(np.int16)
        
        if channels > 1:
            # 创建多通道音频
            audio_data_np = np.stack([audio_data_np] * channels, axis=0)
        
        return audio_data_np.tobytes(), sample_rate, channels

    def test_segment_on_silence(self):
        """
        测试是否能正确地根据静音分割。
        """
        audio_data, sample_rate, channels = self.generate_test_audio()
        
        chunks = segment_audio(
            audio_data=audio_data,
            frame_rate=sample_rate,
            channels=channels,
            max_length_s=12,  # 设置一个较长的最大长度，以确保分割主要由静音决定
            silence_thresh_db=-40
        )
        
        # 预期分割成 3 段 (5s, 5s, 10s)
        self.assertEqual(len(chunks), 3)
        
        # 检查每段的大致长度 (pydub 会增加一些 keep_silence)
        self.assertAlmostEqual(len(chunks[0]) / 1000, 5, delta=0.5)
        self.assertAlmostEqual(len(chunks[1]) / 1000, 5, delta=0.5)
        self.assertAlmostEqual(len(chunks[2]) / 1000, 10, delta=0.5)

    def test_segment_by_max_length(self):
        """
        测试是否能正确地将长片段按最大长度分割。
        """
        audio_data, sample_rate, channels = self.generate_test_audio()
        
        chunks = segment_audio(
            audio_data=audio_data,
            frame_rate=sample_rate,
            channels=channels,
            max_length_s=4,  # 最大长度为 4 秒
            silence_thresh_db=-40
        )
        
        # 预期分割:
        # - 第一段 5s -> 分为 4s, 1s
        # - 第二段 5s -> 分为 4s, 1s
        # - 第三段 10s -> 分为 4s, 4s, 2s
        # 总共 2 + 2 + 3 = 7 段
        self.assertEqual(len(chunks), 7)
        
        # 检查最长片段的长度
        max_chunk_len_s = max(len(c) for c in chunks) / 1000
        self.assertLessEqual(max_chunk_len_s, 4.0)

if __name__ == '__main__':
    unittest.main()
