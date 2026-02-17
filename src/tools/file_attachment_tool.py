"""
File Attachment Tool - Process uploaded files for chat reference.
Supports text files, PDFs, EPUBs, audio, and video files.
"""

import os
import io
import tempfile
import base64
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool


class FileAttachmentTool(BaseTool):
    """Tool to process attached files and extract content for chat context.
    
    Supports:
    - Text files (.txt, .md, .csv, .json, .xml, .py, .js, .html, etc.)
    - PDF files (.pdf)
    - EPUB e-books (.epub)
    - Audio files (.mp3, .wav, .ogg, .m4a, .flac)
    - Video files (.mp4, .avi, .mkv, .mov, .webm)
    
    For audio/video files, attempts transcription using available STT.
    """
    
    # Supported file types
    TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.csv', '.json', '.xml', '.yaml', '.yml', 
                       '.py', '.js', '.ts', '.html', '.htm', '.css', '.sh', '.bash',
                       '.c', '.cpp', '.h', '.hpp', '.java', '.kt', '.go', '.rs',
                       '.rb', '.php', '.swift', '.sql', '.log', '.ini', '.conf'}
    
    PDF_EXTENSIONS = {'.pdf'}
    EPUB_EXTENSIONS = {'.epub'}
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.oga', '.m4a', '.aac', '.flac', '.wma'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.m4v'}
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.upload_dir = Path(config.get('upload_dir', tempfile.gettempdir())) / 'talkie_uploads'
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_text_size = config.get('max_attachment_text_size', 500000)  # 500KB text limit for TTS
        
    def _get_description(self) -> str:
        return (
            "Process attached files (text, PDF, EPUB, audio, video) and extract content for analysis. "
            "Returns the extracted text content and file metadata."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the attached file to process"
                },
                "transcribe_audio": {
                    "type": "boolean",
                    "description": "Whether to transcribe audio/video files (requires STT)",
                    "default": True
                }
            },
            "required": ["file_path"]
        }
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine file type based on extension."""
        ext = Path(file_path).suffix.lower()
        if ext in self.TEXT_EXTENSIONS:
            return 'text'
        elif ext in self.PDF_EXTENSIONS:
            return 'pdf'
        elif ext in self.EPUB_EXTENSIONS:
            return 'epub'
        elif ext in self.AUDIO_EXTENSIONS:
            return 'audio'
        elif ext in self.VIDEO_EXTENSIONS:
            return 'video'
        else:
            return 'unknown'
    
    async def _extract_text_content(self, file_path: str) -> Dict[str, Any]:
        """Extract content from text files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            truncated = len(content) > self.max_text_size
            if truncated:
                content = content[:self.max_text_size] + "\n\n[Content truncated - file too large]"
            
            return {
                "success": True,
                "content": content,
                "content_type": "text",
                "size": len(content),
                "truncated": truncated
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to read text file: {str(e)}"}
    
    async def _extract_pdf_content(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF files."""
        try:
            # Try PyPDF2 first
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text_parts = []
                    total_chars = 0
                    
                    for page_num, page in enumerate(reader.pages):
                        if total_chars >= self.max_text_size:
                            text_parts.append(f"\n\n[PDF truncated at page {page_num + 1} - content too large]")
                            break
                        
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                            total_chars += len(page_text)
                    
                    content = "\n".join(text_parts)
                    
                    return {
                        "success": True,
                        "content": content,
                        "content_type": "pdf_text",
                        "pages": len(reader.pages),
                        "truncated": total_chars >= self.max_text_size
                    }
            except ImportError:
                pass
            
            # Fallback to pdfplumber
            try:
                import pdfplumber
                text_parts = []
                total_chars = 0
                
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        if total_chars >= self.max_text_size:
                            text_parts.append(f"\n\n[PDF truncated at page {page_num + 1} - content too large]")
                            break
                        
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                            total_chars += len(page_text)
                    
                    content = "\n".join(text_parts)
                    
                    return {
                        "success": True,
                        "content": content,
                        "content_type": "pdf_text",
                        "pages": len(pdf.pages),
                        "truncated": total_chars >= self.max_text_size
                    }
            except ImportError:
                return {
                    "success": False,
                    "error": "PDF libraries not installed. Install: pip install PyPDF2 pdfplumber"
                }
                
        except Exception as e:
            return {"success": False, "error": f"Failed to extract PDF content: {str(e)}"}
    
    async def _extract_epub_content(self, file_path: str) -> Dict[str, Any]:
        """Extract text from EPUB files."""
        try:
            from ebooklib import epub
            from bs4 import BeautifulSoup
            
            book = epub.read_epub(file_path)
            text_parts = []
            total_chars = 0
            chapter_count = 0
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapter_count += 1
                    if total_chars >= self.max_text_size:
                        text_parts.append(f"\n\n[EPUB truncated at chapter {chapter_count} - content too large]")
                        break
                    
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    
                    if text:
                        text_parts.append(f"\n--- Chapter {chapter_count} ---\n{text}")
                        total_chars += len(text)
            
            content = "\n".join(text_parts)
            
            return {
                "success": True,
                "content": content,
                "content_type": "epub_text",
                "chapters": chapter_count,
                "title": book.get_metadata('DC', 'title') or ['Unknown'] or [0],
                "truncated": total_chars >= self.max_text_size
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "EPUB libraries not installed. Install: pip install ebooklib beautifulsoup4"
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to extract EPUB content: {str(e)}"}
    
    async def _transcribe_audio(self, file_path: str) -> Dict[str, Any]:
        """Transcribe audio file using available STT."""
        try:
            # Check for whisper availability
            try:
                import whisper
                model = whisper.load_model("base")
                result = model.transcribe(file_path)
                
                return {
                    "success": True,
                    "content": result["text"],
                    "content_type": "audio_transcription",
                    "language": result.get("language", "unknown"),
                    "duration": result.get("duration", 0)
                }
            except ImportError:
                pass
            
            # Check for whisper.cpp via subprocess
            try:
                import subprocess
                result = subprocess.run(
                    ["whisper-cli", "-f", file_path, "--output-txt", "-"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    return {
                        "success": True,
                        "content": result.stdout,
                        "content_type": "audio_transcription"
                    }
            except:
                pass
            
            return {
                "success": False,
                "error": "No speech-to-text engine available. Install whisper or whisper.cpp"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to transcribe audio: {str(e)}"}
    
    async def _process_video(self, file_path: str, transcribe: bool = True) -> Dict[str, Any]:
        """Process video file - extract audio and transcribe if requested."""
        if not transcribe:
            return {
                "success": True,
                "content": "[Video file attached - no transcription requested]",
                "content_type": "video_metadata",
                "metadata": {"transcribed": False}
            }
        
        try:
            # Extract audio from video using ffmpeg
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                audio_path = tmp.name
            
            # Extract audio
            result = subprocess.run([
                "ffmpeg", "-y", "-i", file_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                audio_path
            ], capture_output=True, timeout=120)
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Failed to extract audio from video: {result.stderr.decode()}"
                }
            
            # Transcribe the extracted audio
            transcription = await self._transcribe_audio(audio_path)
            
            # Cleanup
            try:
                os.unlink(audio_path)
            except:
                pass
            
            if transcription["success"]:
                transcription["content_type"] = "video_transcription"
                transcription["metadata"] = {"transcribed": True}
            
            return transcription
            
        except Exception as e:
            return {"success": False, "error": f"Failed to process video: {str(e)}"}
    
    async def execute(self, file_path: str, transcribe_audio: bool = True) -> Dict[str, Any]:
        """Process an attached file and extract content."""
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "content": None
            }
        
        # Get file metadata
        file_stat = os.stat(file_path)
        file_type = self._get_file_type(file_path)
        filename = Path(file_path).name
        
        metadata = {
            "filename": filename,
            "file_type": file_type,
            "size_bytes": file_stat.st_size,
            "path": file_path
        }
        
        # Process based on file type
        if file_type == 'text':
            result = await self._extract_text_content(file_path)
        elif file_type == 'pdf':
            result = await self._extract_pdf_content(file_path)
        elif file_type == 'epub':
            result = await self._extract_epub_content(file_path)
        elif file_type == 'audio':
            if transcribe_audio:
                result = await self._transcribe_audio(file_path)
            else:
                result = {
                    "success": True,
                    "content": f"[Audio file: {filename} - transcription skipped]",
                    "content_type": "audio_metadata"
                }
        elif file_type == 'video':
            result = await self._process_video(file_path, transcribe_audio)
        else:
            result = {
                "success": True,
                "content": f"[Binary file: {filename} - content not extractable]",
                "content_type": "binary"
            }
        
        # Merge metadata into result
        result["metadata"] = metadata
        return result
    
    async def process_upload(self, file_content: bytes, filename: str, transcribe_audio: bool = True) -> Dict[str, Any]:
        """Process an uploaded file from bytes.
        
        Args:
            file_content: Raw bytes of the uploaded file
            filename: Original filename
            transcribe_audio: Whether to transcribe audio/video files
            
        Returns:
            Dict with extracted content and metadata
        """
        # Save to temp file
        file_path = self.upload_dir / filename
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Process the file
        result = await self.execute(str(file_path), transcribe_audio)
        
        # Store the saved path
        result["saved_path"] = str(file_path)
        
        return result
