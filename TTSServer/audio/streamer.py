"""
Audio Streamer - Manages chunked audio streaming
"""

import asyncio
from typing import AsyncGenerator, Optional
from .formats import SAMPLE_RATE, pcm_to_wav


class AudioStreamer:
    """
    Manages audio chunk streaming with buffering.
    """
    
    def __init__(
        self,
        chunk_size: int = 4096,
        sample_rate: int = SAMPLE_RATE,
        buffer_chunks: int = 2
    ):
        """
        Initialize audio streamer.
        
        Args:
            chunk_size: Size of each audio chunk in bytes
            sample_rate: Audio sample rate
            buffer_chunks: Number of chunks to buffer before streaming
        """
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.buffer_chunks = buffer_chunks
        self._buffer = bytearray()
        self._total_bytes = 0
    
    def add_audio(self, pcm_data: bytes) -> list[bytes]:
        """
        Add PCM audio data and return ready chunks.
        
        Args:
            pcm_data: PCM audio bytes to add
        
        Returns:
            List of chunks ready to send
        """
        self._buffer.extend(pcm_data)
        self._total_bytes += len(pcm_data)
        
        chunks = []
        while len(self._buffer) >= self.chunk_size:
            chunk = bytes(self._buffer[:self.chunk_size])
            self._buffer = self._buffer[self.chunk_size:]
            chunks.append(chunk)
        
        return chunks
    
    def flush(self) -> Optional[bytes]:
        """
        Get remaining buffered data.
        
        Returns:
            Remaining bytes or None if buffer is empty
        """
        if self._buffer:
            data = bytes(self._buffer)
            self._buffer = bytearray()
            return data
        return None
    
    def get_total_bytes(self) -> int:
        """Get total bytes processed."""
        return self._total_bytes
    
    def get_duration_seconds(self) -> float:
        """Get total duration in seconds."""
        return self._total_bytes / (self.sample_rate * 2)
    
    @staticmethod
    async def chunk_generator(
        audio_data: bytes,
        chunk_size: int = 4096,
        delay_ms: int = 10
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate chunks from complete audio data.
        
        Args:
            audio_data: Complete audio bytes
            chunk_size: Size of each chunk
            delay_ms: Delay between chunks (simulates streaming)
        
        Yields:
            Audio chunks
        """
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            yield chunk
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
    
    @staticmethod
    def split_text_for_streaming(
        text: str,
        max_chars: int = 100,
        min_chars: int = 20
    ) -> list[str]:
        """
        Split text into chunks for streaming synthesis.
        Tries to split at sentence/clause boundaries.
        
        Args:
            text: Text to split
            max_chars: Maximum characters per chunk
            min_chars: Minimum characters per chunk
        
        Returns:
            List of text chunks
        """
        import re
        
        sentences = re.split(r'([.!?。！？]+[\s]*)', text)
        
        chunks = []
        current = ""
        
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            
            if len(current) + len(sentence) <= max_chars:
                current += sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        
        if sentences and len(sentences) % 2 == 1:
            remaining = sentences[-1]
            if len(current) + len(remaining) <= max_chars:
                current += remaining
            else:
                if current:
                    chunks.append(current.strip())
                current = remaining
        
        if current.strip():
            chunks.append(current.strip())
        
        if not chunks:
            chunks = [text]
        
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chars:
                words = chunk.split()
                sub_chunk = ""
                for word in words:
                    if len(sub_chunk) + len(word) + 1 <= max_chars:
                        sub_chunk += (" " if sub_chunk else "") + word
                    else:
                        if sub_chunk:
                            final_chunks.append(sub_chunk)
                        sub_chunk = word
                if sub_chunk:
                    final_chunks.append(sub_chunk)
            else:
                final_chunks.append(chunk)
        
        return [c for c in final_chunks if c.strip()]
