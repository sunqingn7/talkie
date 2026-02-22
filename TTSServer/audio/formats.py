"""
Audio format utilities for PCM and WAV
"""

import struct
from typing import Optional


SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes


def create_wav_header(
    data_size: Optional[int] = None,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    sample_width: int = SAMPLE_WIDTH
) -> bytes:
    """
    Create a WAV file header.
    
    Args:
        data_size: Size of audio data in bytes (None for streaming)
        sample_rate: Audio sample rate
        channels: Number of channels
        sample_width: Bytes per sample (2 = 16-bit)
    
    Returns:
        WAV header bytes (44 bytes)
    """
    if data_size is None:
        data_size = 0xFFFFFFFF - 36  # Max size for streaming
    
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,
        b'data',
        data_size
    )
    
    return header


def pcm_to_wav(pcm_data: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    Convert PCM audio data to WAV format.
    
    Args:
        pcm_data: Raw PCM audio bytes
        sample_rate: Audio sample rate
    
    Returns:
        Complete WAV file bytes
    """
    header = create_wav_header(len(pcm_data), sample_rate)
    return header + pcm_data


def resample_pcm(
    pcm_data: bytes, 
    from_rate: int, 
    to_rate: int = SAMPLE_RATE
) -> bytes:
    """
    Simple PCM resampling using linear interpolation.
    For better quality, use scipy or librosa.
    
    Args:
        pcm_data: PCM audio bytes (16-bit mono)
        from_rate: Source sample rate
        to_rate: Target sample rate
    
    Returns:
        Resampled PCM bytes
    """
    if from_rate == to_rate:
        return pcm_data
    
    import numpy as np
    
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    ratio = to_rate / from_rate
    new_length = int(len(samples) * ratio)
    
    indices = np.linspace(0, len(samples) - 1, new_length)
    resampled = np.interp(indices, np.arange(len(samples)), samples)
    
    return resampled.astype(np.int16).tobytes()
