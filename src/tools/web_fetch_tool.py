"""
Web Fetch Tool - Fetch and parse webpages to extract readable text.
Uses BeautifulSoup for HTML parsing.
"""

import asyncio
import os
import re
import tempfile
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urljoin
from pathlib import Path

from tools import BaseTool

# Language to voice mapping for Edge TTS
LANGUAGE_VOICE_MAP = {
    "zh-cn": "zh-CN-XiaoxiaoNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "ar": "ar-SA-ZariyahNeural",
    "ru": "ru-RU-DariaNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "es": "es-ES-ElviraNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "en": "en-US-AriaNeural",
}


class WebFetchTool(BaseTool):
    """Tool to fetch and parse webpages.

    Features:
    - Fetches webpage content from URL
    - Extracts readable text from HTML
    - Removes ads, navigation, scripts, styles
    - Returns clean text content
    - Can optionally read aloud via TTS
    - Auto-detects language and selects appropriate TTS voice
    """

    # Common unwanted elements to remove
    REMOVE_TAGS = [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "iframe",
        "noscript",
        "svg",
        "button",
        "input",
        "meta",
        "link",
        "br",
        "hr",
    ]

    def _detect_language(self, text: str) -> str:
        """Detect language from text content."""
        # Check for Chinese characters
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh-cn"

        # Check for Japanese
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
            return "ja"

        # Check for Korean
        if re.search(r"[\uac00-\ud7af]", text):
            return "ko"

        # Check for Arabic
        if re.search(r"[\u0600-\u06ff]", text):
            return "ar"

        # Check for Cyrillic (Russian)
        if re.search(r"[\u0400-\u04ff]", text):
            return "ru"

        # Check for French accented characters
        if re.search(r"[àâäéèêëïîôùûüÿœæç]", text, re.IGNORECASE):
            return "fr"

        # Check for German
        if re.search(r"[äöüß]", text):
            return "de"

        # Check for Spanish
        if re.search(r"[áéíóúüñ¿¡]", text):
            return "es"

        # Default to English
        return "en"

    def _get_tts_voice(self, language: str) -> str:
        """Get TTS voice for detected language."""
        return LANGUAGE_VOICE_MAP.get(language, "en-US-AriaNeural")

    # Classes that typically contain unwanted content
    REMOVE_CLASSES = [
        "advertisement",
        "ad",
        "ads",
        "sidebar",
        "comment",
        "comments",
        "navigation",
        "nav",
        "menu",
        "footer",
        "header",
        "social",
        "share",
        "related",
        "recommended",
        "promo",
        "popup",
        "modal",
        "cookie",
        "banner",
        "newsletter",
        "subscribe",
        "login",
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
                    "description": "The URL of the webpage to fetch and parse",
                },
                "read_aloud": {
                    "type": "boolean",
                    "description": "Whether to read the content aloud using TTS",
                    "default": False,
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum characters to extract (for very long pages)",
                    "default": 10000,
                },
            },
            "required": ["url"],
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
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\t+", "\t", text)

        # Simple cleanup - remove lines that are only URLs
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and URL-only lines
            if line and not re.match(r"^https?://", line):
                if len(line) > 5:  # Skip very short lines
                    cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _extract_text_from_html(self, html: str, max_length: int = 50000) -> str:
        """Extract readable text from HTML."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove unwanted tags
            for tag in soup(self.REMOVE_TAGS):
                tag.decompose()

            # Get text from body directly
            text = ""
            if soup.body:
                text = soup.body.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

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

    def _simple_extract(self, html: str, max_length: int = 50000) -> str:
        """Simple extraction without BeautifulSoup."""
        # Remove scripts and styles
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )

        # Get text content
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Clean
        text = self._clean_text(text)

        # Truncate
        if len(text) > max_length:
            text = text[:max_length] + "\n\n[Content truncated...]"

        return text

    def _find_next_page_link(self, html: str, base_url: str) -> Optional[str]:
        """Find next page link in HTML content."""
        try:
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin

            soup = BeautifulSoup(html, "html.parser")

            # Patterns for next page links (Chinese and English)
            next_patterns = [
                "下一頁",
                "下一章",
                "下一页",
                "下一頁",
                "下一讲",
                "next page",
                "next chapter",
                "next",
                "下一部",
                "继续阅读",
                "繼續閱讀",
                "chapter",
                "章",
                "節",
                "节",
            ]

            # Also look for common link text/aria-labels
            link_texts = ["下一", "下一頁", "next", "继续", "繼續"]

            # Find links that might be next page
            for link in soup.find_all("a", href=True):
                link_text = link.get_text().strip().lower()
                href = link.get("href", "")

                # Check link text
                for pattern in next_patterns:
                    if pattern.lower() in link_text:
                        full_url = urljoin(base_url, href)
                        print(f"[WebFetch] Found next page link: {full_url}")
                        return full_url

                # Check href for common next page patterns (but not current page)
                href_lower = href.lower()

                # Skip if it's the same as base URL (current page)
                if href and href != "/" and base_url.endswith(href):
                    continue

                # Look for next page patterns that indicate a DIFFERENT page
                next_href_patterns = [
                    "/2/",
                    "/3/",
                    "/4/",
                    "/5/",  # /page/2/
                    "_2",
                    "_3",
                    "_4",
                    "_5",  # _2 suffix
                    "?page=2",
                    "?page=3",  # query param
                    "next",  # explicit next
                    "/chapter/",
                    "/ch/",  # chapter in path
                ]

                if any(p in href_lower for p in next_href_patterns):
                    if (
                        href
                        and not href.startswith("#")
                        and not href.startswith("javascript")
                    ):
                        full_url = urljoin(base_url, href)
                        # Make sure it's a valid URL and different from current
                        if full_url.startswith("http") and full_url != base_url:
                            print(f"[WebFetch] Found next page link (href): {full_url}")
                            return full_url

            return None

        except Exception as e:
            print(f"[WebFetch] Error finding next page link: {e}")
            return None

    async def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL content with fallback to cloudscraper on block."""
        try:
            import aiohttp

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

            timeout = aiohttp.ClientTimeout(total=30)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url, headers=headers, allow_redirects=True
                ) as response:
                    print(f"[WebFetch] Status: {response.status}, URL: {response.url}")

                    if response.status == 200:
                        content = await response.text()
                        print(f"[WebFetch] Fetched {len(content)} bytes")
                        return content
                    elif response.status in (403, 429):
                        print(
                            f"[WebFetch] Blocked ({response.status}), trying cloudscraper..."
                        )
                        return await self._fetch_with_cloudscraper(url)
                    else:
                        print(f"[WebFetch] HTTP {response.status}")
                        return None
        except Exception as e:
            print(f"[WebFetch] Error fetching URL: {e}")
            return await self._fetch_with_cloudscraper(url)

    async def _fetch_with_cloudscraper(self, url: str) -> Optional[str]:
        """Fetch using cloudscraper to bypass anti-bot protection."""
        try:
            import cloudscraper
            from concurrent.futures import ThreadPoolExecutor

            print(f"[WebFetch] Using cloudscraper for: {url}")

            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True}
            )

            response = scraper.get(url, timeout=30)
            print(f"[WebFetch] cloudscraper status: {response.status_code}")

            if response.status_code == 200:
                content = response.text
                print(f"[WebFetch] cloudscraper fetched {len(content)} bytes")
                return content
            else:
                print(f"[WebFetch] cloudscraper HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"[WebFetch] cloudscraper error: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def _fetch_with_playwright(self, url: str) -> Optional[str]:
        """Fetch using playwright (headless browser) for JS-heavy sites."""
        try:
            from playwright.async_api import async_playwright
            import asyncio

            print(f"[WebFetch] Using playwright for: {url}")

            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                page = await browser.new_page()

                await page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                """)

                try:
                    await page.goto(url, timeout=45000, wait_until="domcontentloaded")
                    await asyncio.sleep(3)

                    content = await page.content()
                    print(f"[WebFetch] playwright fetched {len(content)} bytes")
                    return content
                finally:
                    await browser.close()
        except Exception as e:
            print(f"[WebFetch] playwright error: {e}")
            return None

    async def execute(
        self, url: str, read_aloud: bool = False, max_length: int = 50000
    ) -> Dict[str, Any]:
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
            return {"success": False, "error": "Invalid URL provided", "url": url}

        # Fetch content with fallback chain: aiohttp -> cloudscraper -> playwright
        html = await self._fetch_url(url)

        if not html:
            print("[WebFetch] Trying cloudscraper...")
            html = await self._fetch_with_cloudscraper(url)

        if not html:
            print("[WebFetch] Trying playwright...")
            html = await self._fetch_with_playwright(url)

        if not html:
            return {
                "success": False,
                "error": "Failed to fetch webpage. The URL may be inaccessible, blocked by anti-bot protection, or requires authentication.",
                "url": url,
            }

        if len(html) < 200:
            print(
                f"[WebFetch] WARNING: Very short response ({len(html)} chars): {html[:200]}"
            )

        # Extract text
        content = self._extract_text_from_html(html, max_length)

        if not content or len(content) < 50:
            return {
                "success": False,
                "error": "Could not extract readable content from this webpage",
                "url": url,
            }

        # Check for pagination and follow if found
        pagination_indicators = [
            "下一頁",
            "下一章",
            "下一页",
            "下一頁",
            "下一部",
            "繼續閱讀",
            "继续阅读",
            "下一讲",
        ]
        has_pagination = any(
            indicator in content for indicator in pagination_indicators
        )

        if has_pagination:
            print(f"[WebFetch] Detected pagination, looking for next page link...")
            next_url = self._find_next_page_link(html, url)

            if next_url and next_url != url:
                print(f"[WebFetch] Fetching next page: {next_url}")
                # Fetch next page
                next_html = await self._fetch_url(next_url)
                if not next_html:
                    next_html = await self._fetch_with_cloudscraper(next_url)
                if not next_html:
                    next_html = await self._fetch_with_playwright(next_url)

                if next_html:
                    next_content = self._extract_text_from_html(next_html, max_length)
                    if next_content and len(next_content) > 50:
                        # Append next page content
                        content = content + "\n\n" + next_content
                        print(f"[WebFetch] Combined content: {len(content)} chars")

        # Try to extract title
        title = ""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except:
            pass

        if not title:
            try:
                title_match = re.search(
                    r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE
                )
                if title_match:
                    title = title_match.group(1).strip()
            except:
                pass

        word_count = len(content.split())

        # Detect language from content
        detected_language = self._detect_language(content)
        tts_voice = self._get_tts_voice(detected_language)

        print(
            f"✅ Fetched {len(content)} chars, {word_count} words, language: {detected_language}, voice: {tts_voice}"
        )

        result = {
            "success": True,
            "url": url,
            "title": title,
            "content": content,
            "word_count": word_count,
            "char_count": len(content),
            "detected_language": detected_language,
            "tts_voice": tts_voice,
        }

        # If read_aloud is requested, save content to temp file for reading
        if read_aloud:
            print(f"[WebFetch] read_aloud=True, saving to temp file...")

            # Save content to temp file
            try:
                # Create temp file with meaningful name
                safe_title = "".join(
                    c for c in (title or "webpage") if c.isalnum() or c in " -_"
                )[:50]
                temp_fd, temp_path = tempfile.mkstemp(
                    suffix=".txt", prefix=f"webfetch_{safe_title}_"
                )

                with os.fdopen(temp_fd, "w") as f:
                    # Write title as header
                    if title:
                        f.write(f"{title}\n\n")
                    f.write(content)

                result["temp_file"] = temp_path
                result["message"] = (
                    f"Saved webpage content ({detected_language}, {word_count} words) to temporary file. Use read_file_chunk to read it with voice: {tts_voice}"
                )
                print(f"[WebFetch] Saved to temp file: {temp_path}")

            except Exception as e:
                print(f"[WebFetch] Error saving to temp file: {e}")
                result["message"] = f"Content fetched but could not save to file: {e}"

        # If voice_daemon is available, also queue for reading with detected language
        if read_aloud and self.voice_daemon:
            print(
                f"[WebFetch] Queueing content for reading via voice daemon (language: {detected_language}, voice: {tts_voice})..."
            )

            # Split content into chunks for reading
            chunks = []
            current_chunk = ""

            for paragraph in content.split("\n\n"):
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

            # Queue chunks for reading with detected language
            for i, chunk in enumerate(chunks):
                self.voice_daemon.speak_file_content(
                    text=chunk,
                    paragraph_num=i,
                    language=detected_language,  # Use detected language
                )

            result["is_reading"] = True
            print(f"[WebFetch] Queued {len(chunks)} chunks for reading")

        return result

    async def fetch_and_read(
        self, url: str, voice_daemon=None, language: str = "auto"
    ) -> Dict[str, Any]:
        """Fetch webpage and read it aloud via Voice Daemon.

        Args:
            url: URL to fetch
            voice_daemon: Voice daemon for TTS
            language: Language code

        Returns:
            Dict with success status and message
        """
        # Fetch content - use larger limit for reading aloud
        result = await self.execute(url, read_aloud=False, max_length=50000)

        if not result.get("success"):
            return result

        content = result["content"]
        title = result.get("title", "Webpage")

        # Queue for reading if voice daemon available
        if voice_daemon:
            # Split into chunks for reading
            chunks = []
            current_chunk = ""

            for paragraph in content.split("\n\n"):
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
                    text=chunk, paragraph_num=i, language=language
                )

            return {
                "success": True,
                "message": f"Reading {len(chunks)} chunks from {title}",
                "url": url,
                "title": title,
                "chunks": len(chunks),
                "word_count": result["word_count"],
            }
        else:
            return {
                "success": True,
                "message": "Fetched but voice daemon not available for reading",
                "url": url,
                "title": title,
                "content": content,
            }
