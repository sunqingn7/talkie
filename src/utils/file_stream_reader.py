import os
from pathlib import Path
from typing import Optional, Tuple


def read_words_from_file(
    file_path: str, start_word: int, count: int
) -> Tuple[str, int]:
    """
    Read 'count' words starting from 'start_word' index.

    Args:
        file_path: Path to the file to read
        start_word: Starting word index (0-based)
        count: Number of words to read

    Returns:
        Tuple of (extracted_text, actual_word_count)
    """
    if not os.path.exists(file_path):
        return "", 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[FileStreamReader] Failed to read file {file_path}: {e}")
        return "", 0

    words = content.split()
    total_words = len(words)

    if start_word >= total_words:
        return "", 0

    end_word = min(start_word + count, total_words)
    extracted = words[start_word:end_word]

    return " ".join(extracted), len(extracted)


def count_total_words(file_path: str) -> int:
    """
    Count total words/chars in a file.
    Now uses true streaming reader.
    """
    from utils.file_reader import count_total_words as _count_total

    return _count_total(file_path)


def read_file_range(file_path: str, start_byte: int, end_byte: int) -> str:
    """
    Read a byte range from a file.

    Args:
        file_path: Path to the file
        start_byte: Starting byte position
        end_byte: Ending byte position

    Returns:
        Extracted text
    """
    if not os.path.exists(file_path):
        return ""

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.seek(start_byte)
            return f.read(end_byte - start_byte)
    except Exception as e:
        print(f"[FileStreamReader] Failed to read range from {file_path}: {e}")
        return ""


def get_file_info(file_path: str) -> dict:
    """
    Get file metadata.

    Args:
        file_path: Path to the file

    Returns:
        Dict with file info (size, mtime, exists)
    """
    if not os.path.exists(file_path):
        return {"exists": False, "size": 0, "mtime": None, "word_count": 0}

    stat = os.stat(file_path)
    word_count = count_total_words(file_path)

    return {
        "exists": True,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "word_count": word_count,
    }


def split_into_chunks(
    text: str, chunk_size: int = 100, min_chunk_size: int = 50
) -> list:
    """
    Split text into chunks based on sentences.
    - Each chunk contains one or more complete sentences
    - Minimum chunk size: min_chunk_size (50 chars/words)
    - No maximum size - sentences are kept together

    Args:
        text: Text to split
        chunk_size: Target size (unused now, kept for compatibility)
        min_chunk_size: Minimum size (default 50 chars for Chinese, words for English)

    Returns:
        List of text chunks
    """
    is_chinese = len(text) > 0 and text.count(" ") < len(text) * 0.1

    if is_chinese:
        return _split_chinese_by_sentences(text, min_chunk_size)
    else:
        return _split_english_by_sentences(text, min_chunk_size)


def _split_chinese_by_sentences(text: str, min_chunk_size: int = 50) -> list:
    """
    Chinese chunking - find delimiter closest to target size.
    Minimum chunk size is 15 chars - never split smaller than that.
    """
    if not text:
        return []

    chunks = []
    pos = 0
    text_len = len(text)

    CHINESE_EOS = "。！？"
    CHINESE_CLAUSE = "，"
    NEWLINE = "\n"

    # Minimum is always 2 for Chinese (to capture short sentences like "墨畫！")
    effective_min = max(min_chunk_size, 2)
    max_chunk_size = min_chunk_size * 3  # Hard limit to prevent huge chunks

    while pos < text_len:
        remaining = text_len - pos

        # If remaining is small, take all
        if remaining <= effective_min:
            chunk = text[pos:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # If remaining exceeds max, limit it
        if remaining > max_chunk_size:
            remaining = max_chunk_size

        # Search range
        search_end = min(pos + remaining, text_len)

        # Find all delimiters in range
        clause_starts = [
            i + 1 for i in range(pos, search_end) if text[i] in CHINESE_CLAUSE
        ]
        eos_starts = [i + 1 for i in range(pos, search_end) if text[i] in CHINESE_EOS]
        newline_starts = [i + 1 for i in range(pos, search_end) if text[i] in NEWLINE]

        split_pos = -1

        # Find delimiter closest to target but >= minimum
        all_delims = clause_starts + eos_starts + newline_starts
        if all_delims:
            # Find one that's at least min_size away
            candidates = [d for d in all_delims if d - pos >= effective_min]
            if candidates:
                split_pos = min(candidates, key=lambda x: abs(x - pos - min_chunk_size))
            else:
                # All are too small, find the largest one
                split_pos = max(all_delims)

        # Fallback: take rest (but not more than max_chunk_size)
        if split_pos == -1 or split_pos <= pos:
            # Force split at max_chunk_size to prevent huge chunks
            split_pos = min(pos + max_chunk_size, text_len)

        # If chunk would be too small, include it anyway (don't skip short sentences)
        # Even tiny sentences like "墨畫！" should be read
        chunk_size = split_pos - pos
        if chunk_size < effective_min and chunk_size > 0:
            # Still use this chunk - don't skip short sentences
            pass

        chunk = text[pos:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        pos = split_pos

    return chunks if chunks else [text] if text.strip() else []


def _split_english_by_sentences(text: str, min_chunk_size: int = 50) -> list:
    """
    Split English text by sentences:
    - Each chunk contains one or more complete sentences
    - If sentences are shorter than min_chunk_size, append more sentences/clauses
    - Uses ., !, ? as primary EOS, and , ; : as secondary break points
    """
    import re

    # Split by sentence end (.!?) first, then by clause separators
    # Primary: . ! ?
    # Secondary: , ; :
    primary_split = re.split(r"(?<=[.!?])\s+", text)

    sentences = []
    for segment in primary_split:
        # Further split by clause separators if segment is too long
        if len(segment) > min_chunk_size * 1.5:
            # Split by , ; :
            clauses = re.split(r"(?<=[,;:])\s+", segment)
            sentences.extend([s.strip() for s in clauses if s.strip()])
        else:
            sentences.append(segment.strip())

    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())

        # If this is the first sentence in a chunk, always add it
        if not current_chunk:
            current_chunk.append(sentence)
            current_len = sentence_len
            continue

        # If adding this sentence would make chunk >= min, add it and finalize
        if current_len + sentence_len >= min_chunk_size:
            current_chunk.append(sentence)
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        else:
            # Too small, keep adding
            current_chunk.append(sentence)
            current_len += sentence_len

    # Don't forget last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def read_chunks_from_file(
    file_path: str, start_pos: int, chunk_size: int = 50, num_chunks: int = 5
) -> Tuple[list, int, int]:
    """
    Read multiple chunks from a file using true streaming.

    Args:
        file_path: Path to the file
        start_pos: Starting position (char for Chinese, word for English)
        chunk_size: Size per chunk
        num_chunks: Number of chunks to read

    Returns:
        Tuple of (chunks list, actual_start_pos, actual_end_pos)
    """
    from utils.file_reader import get_file_reader

    total = count_total_words(file_path)

    if start_pos >= total:
        return [], start_pos, start_pos

    # Read a larger chunk and then split by sentences
    # This ensures we can find sentence boundaries
    with get_file_reader(file_path) as reader:
        reader.seek(start_pos)
        # Read enough to get multiple chunks worth
        read_size = chunk_size * num_chunks * 2  # Read more, split later
        actual_read_size = min(read_size, total - start_pos)
        text = reader.read(actual_read_size)
        print(
            f"[file_stream_reader] read_chunks_from_file: file={file_path}, start_pos={start_pos}, total={total}, read_size={read_size}, actual_read={actual_read_size}, text_len={len(text)}"
        )

    if not text:
        print(f"[file_stream_reader] No text read, returning empty")
        return [], start_pos, start_pos

    # Now split the text by sentences
    chunks = split_into_chunks(text, chunk_size)
    print(f"[file_stream_reader] split_into_chunks returned {len(chunks)} chunks")

    # Take only num_chunks chunks
    chunks = chunks[:num_chunks]
    print(f"[file_stream_reader] after slicing: {len(chunks)} chunks")

    # Calculate actual end position
    actual_end = start_pos + sum(len(c) for c in chunks)

    return chunks, start_pos, actual_end
