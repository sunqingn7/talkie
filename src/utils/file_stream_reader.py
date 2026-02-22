import os
from pathlib import Path
from typing import Optional, Tuple


def read_words_from_file(file_path: str, start_word: int, count: int) -> Tuple[str, int]:
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
        with open(file_path, 'r', encoding='utf-8') as f:
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
    
    return ' '.join(extracted), len(extracted)


def count_total_words(file_path: str) -> int:
    """
    Count total words in a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Total word count
    """
    if not os.path.exists(file_path):
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return len(content.split())
    except Exception as e:
        print(f"[FileStreamReader] Failed to count words in {file_path}: {e}")
        return 0


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
        with open(file_path, 'r', encoding='utf-8') as f:
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
        return {
            'exists': False,
            'size': 0,
            'mtime': None,
            'word_count': 0
        }
    
    stat = os.stat(file_path)
    word_count = count_total_words(file_path)
    
    return {
        'exists': True,
        'size': stat.st_size,
        'mtime': stat.st_mtime,
        'word_count': word_count
    }


def split_into_chunks(text: str, chunk_size: int = 100, min_chunk_size: int = 25) -> list:
    """
    Split text into chunks of approximately chunk_size words.
    For Chinese text (no spaces), uses character count instead.
    Short chunks are merged with the next chunk to meet min_chunk_size.
    
    Args:
        text: Text to split
        chunk_size: Words per chunk (or chars for Chinese)
        min_chunk_size: Minimum words/chars per chunk (merges short chunks with next)
        
    Returns:
        List of text chunks
    """
    # Detect if text is Chinese (no spaces between characters)
    is_chinese = len(text) > 0 and text.count(' ') < len(text) * 0.1
    
    if is_chinese:
        # For Chinese: split by characters
        chars = list(text)
        chunks = []
        current_chunk = []
        current_count = 0
        
        for char in chars:
            current_chunk.append(char)
            current_count += 1
            
            # For Chinese, ONLY split at major sentence-ending punctuation (。！？)
            # NOT at commas (，) or other minor punctuation
            if char in '。！？' or current_count >= chunk_size:
                chunk_text = ''.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                current_chunk = []
                current_count = 0
        
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
    else:
        # For English/European languages: split by words
        words = text.split()
        chunks = []
        current_chunk = []
        current_count = 0
        
        for word in words:
            current_chunk.append(word)
            current_count += 1
            
            if current_count >= chunk_size:
                if word.endswith(('.', '!', '?')) or current_count >= chunk_size + 20:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_count = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    # Merge short chunks with the next chunk to meet min_chunk_size
    if min_chunk_size > 0 and len(chunks) > 1:
        merged_chunks = []
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            current_len = len(current_chunk) if is_chinese else len(current_chunk.split())
            
            # If current chunk is too short, try to merge with next chunk
            if current_len < min_chunk_size and i + 1 < len(chunks):
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged = current_chunk + next_chunk
                merged_chunks.append(merged)
                i += 2  # Skip both chunks
            else:
                merged_chunks.append(current_chunk)
                i += 1
        
        chunks = merged_chunks
    
    return chunks


def read_chunks_from_file(
    file_path: str, 
    start_word: int, 
    chunk_size: int = 100, 
    num_chunks: int = 3
) -> Tuple[list, int, int]:
    """
    Read multiple chunks from a file starting at a given word position.
    
    Args:
        file_path: Path to the file
        start_word: Starting word index
        chunk_size: Words per chunk
        num_chunks: Number of chunks to read
        
    Returns:
        Tuple of (chunks list, actual_start_word, actual_end_word)
    """
    total_words = count_total_words(file_path)
    
    if start_word >= total_words:
        return [], start_word, start_word
    
    words_needed = chunk_size * num_chunks
    end_word = min(start_word + words_needed, total_words)
    
    text, actual_count = read_words_from_file(file_path, start_word, words_needed)
    
    if not text:
        return [], start_word, start_word
    
    chunks = split_into_chunks(text, chunk_size)
    
    actual_end_word = start_word + actual_count
    
    return chunks, start_word, actual_end_word
