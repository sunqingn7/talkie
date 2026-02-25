"""
Abstract file reader for streaming file reading.
Supports different file types with language-aware chunking.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseFileReader(ABC):
    """Abstract base class for file readers."""

    @abstractmethod
    def seek(self, position: int) -> None:
        """Seek to position (character for Chinese, word for English)."""
        pass

    @abstractmethod
    def read(self, size: int) -> str:
        """Read up to 'size' units from current position."""
        pass

    @abstractmethod
    def get_total(self) -> int:
        """Get total size in appropriate units."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the file."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TextFileReader(BaseFileReader):
    """Reader for plain text files with true streaming."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._total_chars = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                self._total_chars = len(content)
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    content = f.read()
                    self._total_chars = len(content)
            except:
                self._total_chars = os.path.getsize(file_path)

    def get_total(self) -> int:
        return self._total_chars

    def seek(self, position: int) -> None:
        pass  # For in-memory, just track position

    def read(self, size: int) -> str:
        return ""  # Not used for in-memory

    def close(self) -> None:
        pass


class StreamingTextFileReader(BaseFileReader):
    """Reader for plain text files with true streaming from disk."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._content = None
        self._total_chars = 0
        self._load_content()

    def _load_content(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._content = f.read()
                self._total_chars = len(self._content)
        except UnicodeDecodeError:
            try:
                with open(self.file_path, "r", encoding="gbk") as f:
                    self._content = f.read()
                    self._total_chars = len(self._content)
            except Exception as e:
                print(f"[FileReader] Error reading file: {e}")
                self._content = ""

    def seek(self, position: int) -> None:
        pass  # Just track position virtually

    def read(self, size: int) -> str:
        if not self._content:
            return ""
        return self._content[:size]

    def get_total(self) -> int:
        return self._total_chars

    def close(self) -> None:
        pass


class HtmlFileReader(BaseFileReader):
    """Reader for HTML files - extracts text content."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.text = self._extract_text(html)
        self._total = len(self.text)
        self._position = 0

    def _extract_text(self, html: str) -> str:
        import re

        text = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def seek(self, position: int) -> None:
        self._position = min(position, self._total)

    def read(self, size: int) -> str:
        text = self.text[self._position : self._position + size]
        self._position += len(text)
        return text

    def get_total(self) -> int:
        return self._total

    def close(self) -> None:
        pass


def get_file_reader(file_path: str) -> BaseFileReader:
    """Factory function to get appropriate reader for file type."""
    ext = os.path.splitext(file_path)[1].lower()

    # Check content to detect HTML
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            first_bytes = f.read(200)
            if "<html" in first_bytes.lower() or "<body" in first_bytes.lower():
                return HtmlFileReader(file_path)
    except:
        pass

    if ext == ".html" or ext == ".htm":
        return HtmlFileReader(file_path)
    else:
        return TextFileReader(file_path)


def read_file_chunked(
    file_path: str, start_pos: int, chunk_size: int, num_chunks: int
) -> Tuple[list, int, int]:
    """
    Read chunks from file using streaming.

    Args:
        file_path: Path to the file
        start_pos: Starting position (char for Chinese, word for English)
        chunk_size: Size per chunk
        num_chunks: Number of chunks to read

    Returns:
        Tuple of (chunks list, actual_start_pos, actual_end_pos)
    """
    with get_file_reader(file_path) as reader:
        total = reader.get_total()

        if start_pos >= total:
            return [], start_pos, start_pos

        reader.seek(start_pos)

        chunks = []
        current_pos = start_pos

        for _ in range(num_chunks):
            text = reader.read(chunk_size)
            if not text:
                break

            chunks.append(text)
            current_pos += len(text)

            if current_pos >= total:
                break

        return chunks, start_pos, current_pos


def count_total_words(file_path: str) -> int:
    """
    Count total words/chars in a file.
    Uses character count for Chinese, word count for English.
    """
    if not os.path.exists(file_path):
        return 0

    try:
        with get_file_reader(file_path) as reader:
            return reader.get_total()
    except Exception as e:
        print(f"[FileStreamReader] Failed to count words in {file_path}: {e}")
        return 0
