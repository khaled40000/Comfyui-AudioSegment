import unittest
import os
import sys
from pydub import AudioSegment

# 将项目根目录添加到 Python 路径中，以便导入模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from audio_paragraph_segmenter import segment_by_paragraph

class TestParagraphSegmenter(unittest.TestCase):

    def setUp(self):
        """设置测试环境"""
        self.audio_path = "/Users/cherokee/Downloads/caocao.wav"
        self.text_path = "/Users/cherokee/Downloads/caocao.txt"
        self.output_dir = "output"
        self.model_name = "base"
        
        # ComfyUI 的模型目录结构
        # 为了测试，我们假设模型会自动下载到默认位置，因此 model_dir 设为 None 或一个临时目录
        # Whisper 库会自动处理下载
        self.model_dir = os.path.expanduser("~/.cache/whisper")

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 检查输入文件是否存在
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"测试音频文件未找到: {self.audio_path}")
        if not os.path.exists(self.text_path):
            raise FileNotFoundError(f"测试文本文件未找到: {self.text_path}")

    def test_segmentation_and_srt_output(self):
        """测试音频分割和 SRT 字幕生成功能"""
        # 1. 读取音频文件
        audio = AudioSegment.from_wav(self.audio_path)
        audio_data = audio.raw_data
        sample_rate = audio.frame_rate
        channels = audio.channels

        # 2. 读取文本文件
        with open(self.text_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # 3. 调用核心分割函数
        audio_clips, count, srt_string = segment_by_paragraph(
            audio_data=audio_data,
            sample_rate=sample_rate,
            channels=channels,
            text=text_content,
            model_name=self.model_name,
            model_dir=self.model_dir
        )

        # 4. 断言结果不为空
        self.assertIsNotNone(audio_clips)
        self.assertGreater(count, 0)
        self.assertIsNotNone(srt_string)
        self.assertTrue(len(srt_string) > 0)
        self.assertEqual(count, len(audio_clips))

        print(f"成功分割成 {count} 个片段。")
        print("SRT 内容预览:")
        print(srt_string[:200] + "...")

        # 5. 保存输出文件
        # 保存 SRT
        srt_output_path = os.path.join(self.output_dir, "caocao.srt")
        with open(srt_output_path, 'w', encoding='utf-8') as f:
            f.write(srt_string)
        print(f"SRT 字幕文件已保存到: {srt_output_path}")

        # 保存音频片段
        for i, clip in enumerate(audio_clips):
            segment_audio = clip["audio"]
            segment_path = os.path.join(self.output_dir, f"segment_{i}.wav")
            segment_audio.export(segment_path, format="wav")
        
        print(f"{count} 个音频片段已保存到 '{self.output_dir}' 目录。")

if __name__ == '__main__':
    unittest.main()
