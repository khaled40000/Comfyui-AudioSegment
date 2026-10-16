"""
ComfyUI-AudioSegment
"""

from .nodes import (
    AudioSegmenterNode, 
    SelectAudioFromList, 
    AudioListToBatch, 
    AudioSegmentByParagraphNode,
    SelectAudioClipFromList
)

NODE_CLASS_MAPPINGS = {
    "AudioSegmenter": AudioSegmenterNode,
    "AudioSegmentByParagraph": AudioSegmentByParagraphNode,
    "SelectAudioFromList": SelectAudioFromList,
    "SelectAudioClipFromList": SelectAudioClipFromList,
    "AudioListToBatch": AudioListToBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSegmenter": "Audio Segmenter",
    "AudioSegmentByParagraph": "Audio Segment By Paragraph",
    "SelectAudioFromList": "Select Audio From List",
    "SelectAudioClipFromList": "Select AudioClip From List",
    "AudioListToBatch": "Audio List to Batch",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
