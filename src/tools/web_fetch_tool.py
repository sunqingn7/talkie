"""
Web Fetch Tool - Fetch and parse webpages to extract readable text.
Uses BeautifulSoup for HTML parsing.
"""

import asyncio
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from tools import BaseTool


class WebFetchTool(BaseTool):
    """Tool to fetch and parse webpages.
    
    Features:
    - Fetches webpage content from URL
    - Extracts readable text from HTML
    - Removes ads, navigation, scripts, styles
    - Returns clean text content
    - Can optionally read aloud via TTS
    """
    
    # Common unwanted elements to remove
    REMOVE_TAGS = [
        'script', 'style', 'nav', 'header', 'footer', 'aside',
        'form', 'iframe', 'noscript', 'svg', 'button', 'input',
        'meta', 'link', 'br', 'hr'
    ]
    
    # Classes that typically contain unwanted content
    REMOVE_CLASSES = [
        'advertisement', 'ad', 'ads', 'sidebar', 'comment', 'comments',
        'navigation', 'nav', 'menu', 'footer', 'header', 'social',
        'share', 'related', 'recommended', 'promo', 'popup', 'modal',
        'cookie', 'banner', 'newsletter', 'subscribe', 'login'
    ]
    
    def _get_description(self) -> str:
        return (
            "Fetch and read webpage content from a URL. "
            "Extracts clean, readable text from HTML pages, removing ads, "
            "navigation, and other non-content elements. "
            "Can read the content aloud if requested."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to fetch and parse"
                },
                "read_aloud": {
                    "type": "boolean",
                    "description": "Whether to read the content aloud using TTS",
                    "default": False
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum characters to extract (for very long pages)",
                    "default": 10000
                }
            },
            "required": ["url"]
        }
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        self.voice_daemon = None  # Will be set by MCP server
    
    def set_voice_daemon(self, voice_daemon):
        """Set the voice daemon for reading aloud."""
        self.voice_daemon = voice_daemon
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\t+', '\t', text)
        
        # Simple cleanup - remove lines that are only URLs
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and URL-only lines
            if line and not re.match(r'^https?://', line):
                if len(line) > 5:  # Skip very short lines
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_text_from_html(self, html: str, max_length: int = 10000) -> str:
        """Extract readable text from HTML."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted tags
            for tag in soup(self.REMOVE_TAGS):
                tag.decompose()
            
            # Get text from body directly
            text = ""
            if soup.body:
                text = soup.body.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            print(f"[WebFetch] Raw text length: {len(text)}")
            
            # Clean the text
            text = self._clean_text(text)
            
            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length] + "\n\n[Content truncated...]"
            
            return text
            
        except ImportError:
            # Fallback: simple regex-based extraction
            print("[WebFetch] BeautifulSoup not available, using simple extraction")
            return self._simple_extract(html, max_length)
        except Exception as e:
            import traceback
            print(f"[WebFetch] Error parsing HTML: {e}")
            traceback.print_exc()
            return f"Error parsing HTML: {str(e)}"
    
    def _simple_extract(self, html: str, max_length: int = 10000) -> str:
        """Simple extraction without BeautifulSoup."""
        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Get text content
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Clean
        text = self._clean_text(text)
        
        # Truncate
        if len(text) > max_length:
            text = text[:max_length] + "\n\n[Content truncated...]"
        
        return text
    
    async def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL content."""
        try:
            import aiohttp
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    print(f"[WebFetch] Status: {response.status}, URL: {response.url}")
                    
                    if response.status == 200:
                        content = await response.text()
                        print(f"[WebFetch] Fetched {len(content)} bytes")
                        return content
                    else:
                        print(f"[WebFetch] HTTP {response.status}")
                        return None
        except Exception as e:
            print(f"[WebFetch] Error fetching URL: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def execute(self, url: str, read_aloud: bool = False, 
                     max_length: int = 10000) -> Dict[str, Any]:
        """Fetch and parse a webpage.
        
        Args:
            url: URL to fetch
            read_aloud: Whether to read content via TTS
            max_length: Maximum characters to extract
            
        Returns:
            Dict with success, title, content, word_count, url
        """
        print(f"🌐 Fetching webpage: {url}")
        
        # Validate URL
        if not self._is_valid_url(url):
            return {
                "success": False,
                "error": "Invalid URL provided",
                "url": url
            }
        
        # Fetch content
        html = await self._fetch_url(url)
        
        if not html:
            return {
                "success": False,
                "error": "Failed to fetch webpage. The URL may be inaccessible or requires authentication.",
                "url": url
            }
        
        if len(html) < 200:
            print(f"[WebFetch] WARNING: Very short response ({len(html)} chars): {html[:200]}")
        
        # Extract text
        content = self._extract_text_from_html(html, max_length)
        
        if not content or len(content) < 50:
            return {
                "success": False,
                "error": "Could not extract readable content from this webpage",
                "url": url
            }
        
        # Try to extract title
        title = ""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except:
            pass
        
        if not title:
            try:
                title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
            except:
                pass
        
        word_count = len(content.split())
        
        print(f"✅ Fetched {len(content)} chars, {word_count} words")
        
        result = {
            "success": True,
            "url": url,
            "title": title,
            "content": content,
            "word_count": word_count,
            "char_count": len(content)
        }
        
        # If read_aloud is requested, read the content via VoiceDaemon
        if read_aloud and self.voice_daemon:
            print(f"[WebFetch] read_aloud=True, queuing content for reading...")
            
            # Split content into chunks for reading
            chunks = []
            current_chunk = ""
            
            for paragraph in content.split('\n\n'):
                if len(current_chunk) + len(paragraph) > 800:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = paragraph[:800]
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Queue chunks for reading
            for i, chunk in enumerate(chunks):
                self.voice_daemon.speak_file_content(
                    text=chunk,
                    paragraph_num=i+1,
                    language=language
                )
            
            result["message"] = f"Reading {len(chunks)} chunks from {title or 'webpage'}"
            result["is_reading"] = True
            print(f"[WebFetch] Queued {len(chunks)} chunks for reading")
        
        return result
    
    async def fetch_and_read(self, url: str, voice_daemon=None, 
                           language: str = "auto") -> Dict[str, Any]:
        """Fetch webpage and read it aloud via Voice Daemon.
        
        Args:
            url: URL to fetch
            voice_daemon: Voice daemon for TTS
            language: Language code
            
        Returns:
            Dict with success status and message
        """
        # Fetch content
        result = await self.execute(url, read_aloud=False, max_length=15000)
        
        if not result.get("success"):
            return result
        
        content = result["content"]
        title = result.get("title", "Webpage")
        
        # Queue for reading if voice daemon available
        if voice_daemon:
            # Split into chunks for reading
            chunks = []
            current_chunk = ""
            
            for paragraph in content.split('\n\n'):
                if len(current_chunk) + len(paragraph) > 800:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = paragraph[:800]
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Queue chunks
            for i, chunk in enumerate(chunks):
                voice_daemon.speak_file_content(
                    text=chunk,
                    paragraph_num=i+1,
                    language=language
                )
            
            return {
                "success": True,
                "message": f"Reading {len(chunks)} chunks from {title}",
                "url": url,
                "title": title,
                "chunks": len(chunks),
                "word_count": result["word_count"]
            }
        else:
            return {
                "success": True,
                "message": "Fetched but voice daemon not available for reading",
                "url": url,
                "title": title,
                "content": content
            }
