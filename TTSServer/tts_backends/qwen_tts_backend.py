"""
Qwen TTS Backend - Local Qwen3-TTS with true streaming support
Uses direct transformers model for low-latency streaming with pyaudio
"""

import asyncio
import hashlib
import os
import re
import tempfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Dict, List, Any, Optional

from .base import TTSBackend
from ..audio.formats import pcm_to_wav, SAMPLE_RATE


# Streaming audio player using pyaudio
class StreamingAudioPlayer:
    """Real-time audio streaming player using pyaudio."""

    def __init__(self, sample_rate: int = 24000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None
        self.paudio = None
        self._initialized = False

    def initialize(self):
        """Initialize pyaudio stream."""
        if self._initialized:
            return

        try:
            import pyaudio
            import numpy as np

            self.paudio = pyaudio.PyAudio()
            self.stream = self.paudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
            )
            self._initialized = True
            print(
                f"[StreamingPlayer] Initialized pyaudio stream at {self.sample_rate}Hz"
            )
        except ImportError:
            print("[StreamingPlayer] pyaudio not available, using file-based streaming")
            self._initialized = False

    def play_chunk(self, audio_data: bytes):
        """Play audio chunk in real-time."""
        if not self._initialized or self.stream is None:
            return

        import numpy as np

        audio = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio.astype(np.float32) / 32768.0

        self.stream.write(audio_float.tobytes())

    def flush(self):
        """Flush remaining audio."""
        if self.stream:
            self.stream.stop_stream()

    def close(self):
        """Close the audio stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.paudio:
            self.paudio.terminate()
            self.paudio = None
        self._initialized = False


class LRUCache:
    """Thread-safe LRU cache for audio segments."""

    def __init__(self, max_size: int = 100, max_bytes: int = 50 * 1024 * 1024):
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.cache: OrderedDict[str, bytes] = OrderedDict()
        self.total_bytes = 0
        self.hits = 0
        self.misses = 0

    def _make_key(self, text: str, speaker: str, language: str) -> str:
        data = f"{text}|{speaker}|{language}"
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, text: str, speaker: str, language: str) -> Optional[bytes]:
        key = self._make_key(text, speaker, language)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, text: str, speaker: str, language: str, audio: bytes):
        key = self._make_key(text, speaker, language)

        if key in self.cache:
            self.cache.move_to_end(key)
            return

        while (
            len(self.cache) >= self.max_size
            or self.total_bytes + len(audio) > self.max_bytes
        ):
            if not self.cache:
                break
            oldest_key, oldest_audio = self.cache.popitem(last=False)
            self.total_bytes -= len(oldest_audio)

        self.cache[key] = audio
        self.total_bytes += len(audio)

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return {
            "size": len(self.cache),
            "bytes": self.total_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


class QwenTTSBackend(TTSBackend):
    """Qwen3-TTS backend with true streaming for low latency.

    Uses direct transformers model with:
    - 12Hz tokenizer for causal generation
    - Flash Attention 2 for 30-40% RTF boost
    - torch.compile for overhead reduction
    - pyaudio for real-time streaming
    """

    name = "qwen_tts"
    description = (
        "Qwen3-TTS - Local neural TTS with multiple voices (offline, streaming)"
    )

    # 12Hz tokenizer and model IDs for streaming
    TOKENIZER_12HZ = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    MODEL_12HZ_0_6B = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    MODEL_12HZ_1_7B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    CUSTOM_VOICE_SPEAKERS = [
        {
            "id": "Vivian",
            "name": "Vivian",
            "language": "zh",
            "gender": "Female",
            "description": "Bright, slightly edgy young female voice",
        },
        {
            "id": "Serena",
            "name": "Serena",
            "language": "zh",
            "gender": "Female",
            "description": "Warm, gentle young female voice",
        },
        {
            "id": "Uncle_Fu",
            "name": "Uncle Fu",
            "language": "zh",
            "gender": "Male",
            "description": "Seasoned male voice with low, mellow timbre",
        },
        {
            "id": "Dylan",
            "name": "Dylan",
            "language": "zh",
            "gender": "Male",
            "description": "Youthful Beijing male voice",
        },
        {
            "id": "Eric",
            "name": "Eric",
            "language": "zh",
            "gender": "Male",
            "description": "Lively Chengdu male voice",
        },
        {
            "id": "Ryan",
            "name": "Ryan",
            "language": "en",
            "gender": "Male",
            "description": "Dynamic male voice with strong rhythmic drive",
        },
        {
            "id": "Aiden",
            "name": "Aiden",
            "language": "en",
            "gender": "Male",
            "description": "Sunny American male voice",
        },
        {
            "id": "Ono_Anna",
            "name": "Ono Anna",
            "language": "ja",
            "gender": "Female",
            "description": "Playful Japanese female voice",
        },
        {
            "id": "Sohee",
            "name": "Sohee",
            "language": "ko",
            "gender": "Female",
            "description": "Warm Korean female voice",
        },
    ]

    SUPPORTED_LANGUAGES = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]

    MODEL_CONFIGS = {
        "custom_voice_0.6b": {
            "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "params": "0.6B",
        },
        "custom_voice_1.7b": {
            "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "params": "1.7B",
        },
        "voice_design_1.7b": {
            "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "params": "1.7B",
        },
        "base_0.6b": {
            "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "params": "0.6B",
        },
        "base_1.7b": {
            "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "params": "1.7B",
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.model_type = self.config.get("model_type", "custom_voice_0.6b")
        self.default_speaker = self.config.get("default_speaker", "Vivian")
        self.device = self.config.get("device", "cuda:0")
        self.dtype = self.config.get("dtype", "bfloat16")

        # Streaming config
        self.use_streaming_mode = self.config.get("use_streaming_mode", True)
        self.chunk_size = self.config.get(
            "chunk_size", 50
        )  # Text chunk size for low latency
        self.streaming_player = None

        # Model components (direct transformers approach)
        self._model = None
        self._tokenizer = None
        self._fast_model = None
        self._fast_tokenizer = None
        self._model_loaded = False
        self._model_loading = False
        self._load_error = None
        self._fast_model_loaded = False
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._load_lock = asyncio.Lock()

        self._cache = LRUCache(max_size=200, max_bytes=100 * 1024 * 1024)

        self._hybrid_mode = self.config.get("hybrid_mode", True)
        self._fast_model_for_first = self.config.get("fast_model_for_first", True)

        # Sample rate for Qwen3-TTS (24kHz)
        self.sample_rate = 24000

    async def initialize(self) -> bool:
        """Initialize backend and start loading model in background."""
        import sys

        print(f"[Qwen TTS Backend] Initializing with streaming mode...", flush=True)
        sys.stdout.flush()

        if self.use_streaming_mode:
            try:
                self.streaming_player = StreamingAudioPlayer(
                    sample_rate=self.sample_rate, chunk_size=1024
                )
                self.streaming_player.initialize()
                print(f"[Qwen TTS Backend] Streaming player initialized", flush=True)
            except Exception as e:
                print(f"[Qwen TTS Backend] Could not initialize streaming player: {e}")
                self.streaming_player = None

        asyncio.create_task(self._load_model_background())

        return True

    async def _load_model_background(self):
        """Load model in background thread using direct transformers approach."""
        import sys

        async with self._load_lock:
            if self._model_loaded or self._model_loading:
                return

            self._model_loading = True
            self._load_error = None

        try:
            import torch
            from transformers import Qwen3TTSForConditionalGeneration, AutoTokenizer

            model_config = self.MODEL_CONFIGS.get(self.model_type)
            if not model_config:
                print(
                    f"[Qwen TTS Backend] Unknown model type: {self.model_type}",
                    flush=True,
                )
                self._load_error = f"Unknown model type: {self.model_type}"
                return

            model_id = model_config["model_id"]

            print(
                f"[Qwen TTS Backend] Loading 12Hz tokenizer and model: {model_id}",
                flush=True,
            )
            sys.stdout.flush()

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

            loop = asyncio.get_event_loop()

            self._tokenizer = await loop.run_in_executor(
                self._executor,
                lambda: AutoTokenizer.from_pretrained(
                    self.TOKENIZER_12HZ, trust_remote_code=True
                ),
            )

            self._model = await loop.run_in_executor(
                self._executor,
                lambda: Qwen3TTSForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    device_map=self.device,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True,
                ).to(self.device),
            )

            self._model = torch.compile(
                self._model, mode="reduce-overhead", fullgraph=True
            )

            self._model_loaded = True
            print(
                f"[Qwen TTS Backend] Model loaded and compiled successfully!",
                flush=True,
            )
            sys.stdout.flush()

            if self._hybrid_mode and "1.7b" in self.model_type:
                await self._load_fast_model()

        except ImportError as e:
            print(f"[Qwen TTS Backend] transformers not installed: {e}", flush=True)
            self._load_error = str(e)
        except Exception as e:
            print(f"[Qwen TTS Backend] Failed to load model: {e}", flush=True)
            import traceback

            traceback.print_exc()
            self._load_error = str(e)
        finally:
            self._model_loading = False

    async def _ensure_model_loaded(self) -> bool:
        """Wait for model to be loaded (started in background)."""
        if self._model_loaded:
            return True

        if self._load_error:
            print(f"[Qwen TTS] Model load failed: {self._load_error}")
            return False

        # Model is loading in background, wait for it
        if self._model_loading:
            print(f"[Qwen TTS] Waiting for model to finish loading...")

            # Wait for model to load (with timeout)
            max_wait = 120  # 2 minutes max
            waited = 0
            while self._model_loading and waited < max_wait:
                await asyncio.sleep(0.5)
                waited += 0.5

            if self._model_loaded:
                return True
            if self._load_error:
                print(f"[Qwen TTS] Model load failed: {self._load_error}")
                return False
            print(f"[Qwen TTS] Model load timed out after {max_wait}s")
            return False

        # Model not loading and not loaded - start it
        await self._load_model_background()
        return self._model_loaded

    async def _load_fast_model(self):
        """Load smaller model for hybrid mode using direct transformers."""
        if self._fast_model_loaded:
            return

        try:
            import torch
            from transformers import Qwen3TTSForConditionalGeneration, AutoTokenizer

            print("[Qwen TTS] Loading fast model (0.6B) for hybrid mode...")

            loop = asyncio.get_event_loop()

            self._fast_tokenizer = await loop.run_in_executor(
                None,
                lambda: AutoTokenizer.from_pretrained(
                    self.TOKENIZER_12HZ, trust_remote_code=True
                ),
            )

            self._fast_model = await loop.run_in_executor(
                None,
                lambda: Qwen3TTSForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True,
                ).to(self.device),
            )

            self._fast_model = torch.compile(
                self._fast_model, mode="reduce-overhead", fullgraph=True
            )
            self._fast_model_loaded = True
            print("[Qwen TTS] Fast model loaded and compiled")

        except Exception as e:
            print(f"[Qwen TTS] Failed to load fast model: {e}")
            import traceback

            traceback.print_exc()
            self._fast_model = None

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False

        if self._fast_model is not None:
            del self._fast_model
            self._fast_model = None
            self._fast_model_loaded = False

        if self._executor:
            self._executor.shutdown(wait=False)

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> AsyncIterator[bytes]:
        """
        Stream audio by synthesizing text in chunks.

        Uses aggressive chunking at phrase boundaries for lower latency.
        """
        if not self._model_loaded:
            if not await self._ensure_model_loaded():
                raise RuntimeError("Qwen TTS model not loaded")

        speaker = voice or self.default_speaker
        lang = self._normalize_language(language)

        chunks = self._split_text_aggressive(text)

        print(
            f"[Qwen TTS] Split into {len(chunks)} chunks: {[c[:30] + '...' if len(c) > 30 else c for c in chunks]}"
        )

        is_first_chunk = True

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            try:
                cached = self._cache.get(chunk, speaker, lang)
                if cached:
                    print(f"[Qwen TTS] Cache hit for chunk {i + 1}: '{chunk[:30]}...'")
                    sub_chunks = self._split_pcm_for_streaming(cached, chunk_size=4096)
                    for sub_chunk in sub_chunks:
                        yield sub_chunk
                    is_first_chunk = False
                    continue

                use_fast_model = (
                    is_first_chunk
                    and self._fast_model_for_first
                    and self._fast_model_loaded
                    and self._fast_model is not None
                )

                if use_fast_model:
                    print(f"[Qwen TTS] Using fast model for first chunk")

                pcm_data = await self._synthesize_chunk(
                    chunk, speaker, lang, use_fast_model=use_fast_model
                )

                if pcm_data:
                    self._cache.put(chunk, speaker, lang, pcm_data)

                    sub_chunks = self._split_pcm_for_streaming(
                        pcm_data, chunk_size=4096
                    )
                    for sub_chunk in sub_chunks:
                        yield sub_chunk

                is_first_chunk = False

            except Exception as e:
                print(f"[Qwen TTS] Error synthesizing chunk: {e}")
                continue

        stats = self._cache.get_stats()
        print(
            f"[Qwen TTS] Cache stats: {stats['hits']} hits, {stats['misses']} misses, {stats['hit_rate']:.1f}% hit rate"
        )

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Generate complete WAV audio."""
        pcm_chunks = []
        async for chunk in self.synthesize_stream(
            text, voice, language, speed, **kwargs
        ):
            pcm_chunks.append(chunk)

        pcm_data = b"".join(pcm_chunks)
        return pcm_to_wav(pcm_data)

    async def _synthesize_chunk(
        self, text: str, speaker: str, language: str, use_fast_model: bool = False
    ) -> Optional[bytes]:
        """Synthesize a single text chunk to PCM audio using direct transformers model."""
        import numpy as np

        model = self._fast_model if use_fast_model else self._model
        tokenizer = self._fast_tokenizer if use_fast_model else self._tokenizer

        if model is None or tokenizer is None:
            if model is None:
                model = self._model
            tokenizer = self._tokenizer

        if model is None or tokenizer is None:
            return None

        try:
            loop = asyncio.get_event_loop()

            def generate_audio():
                import torch

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    text_column="text",
                    speaker_id_column=speaker,
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                    )

                    audio = model.decode(generated_ids)

                    audio_np = audio.cpu().numpy().squeeze()

                    window = np.hanning(len(audio_np))
                    audio_np = audio_np * window

                    audio_int16 = (audio_np * 32767).astype(np.int16)

                    return audio_int16.tobytes()

            pcm_data = await loop.run_in_executor(self._executor, generate_audio)
            return pcm_data

        except Exception as e:
            print(f"[Qwen TTS] Synthesis error: {e}")
            import traceback

            traceback.print_exc()

        return None

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available speakers."""
        return self.CUSTOM_VOICE_SPEAKERS.copy()

    def get_supported_languages(self) -> List[str]:
        """Get supported language codes."""
        return self.SUPPORTED_LANGUAGES.copy()

    def get_cache_stats(self) -> dict:
        """Get audio cache statistics."""
        return self._cache.get_stats()

    def clear_cache(self):
        """Clear the audio cache."""
        self._cache = LRUCache(max_size=200, max_bytes=100 * 1024 * 1024)
        print("[Qwen TTS] Cache cleared")

    def _normalize_language(self, language: Optional[str]) -> str:
        """Normalize language code to Qwen format."""
        if not language:
            return "Auto"

        lang_map = {
            "en": "English",
            "zh": "Chinese",
            "zh-cn": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "de": "German",
            "fr": "French",
            "ru": "Russian",
            "pt": "Portuguese",
            "es": "Spanish",
            "it": "Italian",
        }

        return lang_map.get(language.lower(), "Auto")

    def _split_text_aggressive(
        self, text: str, max_chars: int = 80, min_chars: int = 15
    ) -> List[str]:
        """
        Aggressively split text at phrase boundaries for lower latency.

        Splits at:
        1. Sentence endings (. ! ? 。 ！ ？)
        2. Clause markers (, ; ： ， ；)
        3. Natural phrase boundaries
        """
        text = text.strip()
        if not text:
            return []

        sentence_parts = re.split(r"([.!?。！？]+[\s]*)", text)

        sentences = []
        for i in range(0, len(sentence_parts) - 1, 2):
            sentence = sentence_parts[i] + (
                sentence_parts[i + 1] if i + 1 < len(sentence_parts) else ""
            )
            if sentence.strip():
                sentences.append(sentence.strip())

        if len(sentence_parts) % 2 == 1 and sentence_parts[-1].strip():
            sentences.append(sentence_parts[-1].strip())

        if not sentences:
            sentences = [text]

        chunks = []

        for sentence in sentences:
            if len(sentence) <= max_chars:
                chunks.append(sentence)
                continue

            clause_parts = re.split(r"([,;:，、；：]+[\s]*)", sentence)

            current = ""
            for i in range(0, len(clause_parts) - 1, 2):
                clause = clause_parts[i] + (
                    clause_parts[i + 1] if i + 1 < len(clause_parts) else ""
                )

                if len(current) + len(clause) <= max_chars:
                    current += clause
                else:
                    if current and len(current) >= min_chars:
                        chunks.append(current.strip())
                    elif current:
                        if chunks:
                            chunks[-1] += " " + current.strip()
                        else:
                            chunks.append(current.strip())
                    current = clause

            if len(clause_parts) % 2 == 1 and clause_parts[-1].strip():
                remaining = clause_parts[-1].strip()
                if len(current) + len(remaining) <= max_chars:
                    current += remaining
                else:
                    if current:
                        chunks.append(current.strip())
                    current = remaining

            if current.strip():
                if len(current) >= min_chars:
                    chunks.append(current.strip())
                elif chunks:
                    chunks[-1] += " " + current.strip()
                else:
                    chunks.append(current.strip())

        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final_chunks.append(chunk)
            else:
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

        return [c for c in final_chunks if c.strip()]

    def _split_pcm_for_streaming(
        self, pcm_data: bytes, chunk_size: int = 4096
    ) -> List[bytes]:
        """Split PCM data into chunks for streaming."""
        chunks = []
        for i in range(0, len(pcm_data), chunk_size):
            chunks.append(pcm_data[i : i + chunk_size])
        return chunks

    def _resample_audio(self, audio, from_rate: int, to_rate: int):
        """Resample audio to target sample rate."""
        import numpy as np

        if from_rate == to_rate:
            return audio

        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)

        return resampled.astype(np.int16)
