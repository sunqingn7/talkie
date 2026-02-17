"""
Text-to-Speech Tool - Converts text to spoken audio.
Uses Coqui TTS for high-quality neural voices with multilingual support.
Supports: English, Chinese (中文), and 15+ other languages via XTTS-v2.
Falls back to pyttsx3 if Coqui TTS is not available.
"""

import asyncio
import os
import tempfile
import subprocess
from typing import Any, Dict, Optional
import time
from pathlib import Path
import re

# Fix for PyTorch 2.6+ weights loading - must be done before importing TTS
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

# Patch trainer.io to disable weights_only loading
# This is needed because older Coqui TTS models were saved without weights_only support
try:
    import trainer.io as trainer_io
    trainer_io._WEIGHTS_ONLY = False
except ImportError:
    pass

from . import BaseTool


class TTSTool(BaseTool):
    """Tool to convert text to speech using Coqui TTS (or pyttsx3 fallback).
    
    Supports multiple languages:
    - English (en)
    - Chinese/中文 (zh-cn, zh)
    - And 15+ other languages via XTTS-v2
    """
    
    # Track TTS call count for debugging
    call_count = 0
    
    # Language to model mapping
    LANGUAGE_MODELS = {
        "en": "tts_models/en/ljspeech/tacotron2-DDC",  # English
        "zh": "tts_models/zh-CN/baker/tacotron2-DDC-GST",  # Chinese
        "zh-cn": "tts_models/zh-CN/baker/tacotron2-DDC-GST",  # Chinese (Simplified)
        "multilingual": "tts_models/multilingual/multi-dataset/xtts_v2",  # XTTS-v2 (17 languages)
    }
    
    # Languages supported by XTTS-v2
    XTTS_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", 
        "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi", "zh"
    ]
    
    def _get_description(self) -> str:
        return (
            "Convert text to spoken audio using high-quality neural TTS. "
            "Supports English, Chinese (中文), and 15+ other languages. "
            "Uses Coqui TTS for natural-sounding voices. "
            "Auto-detects language or specify manually."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud (supports English, Chinese 中文, and 15+ languages)"
                },
                "language": {
                    "type": "string",
                    "description": "Language code: 'en' (English), 'zh' or 'zh-cn' (Chinese 中文), 'es' (Spanish), 'fr' (French), etc. Auto-detected if not specified.",
                    "default": "auto"
                },
                "speaker_id": {
                    "type": "string",
                    "description": "Speaker/voice ID for multi-speaker models (e.g., 'Claribel Dervla', 'Damien Black')",
                    "default": None
                },
                "speed": {
                    "type": "number",
                    "description": "Speech speed multiplier (0.5 = slow, 1.0 = normal, 1.5 = fast)",
                    "default": 1.0
                }
            },
            "required": ["text"]
        }
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.tts_config = config.get("tts", {})
        
        # Engine management
        self.current_engine = self.tts_config.get("engine", "edge_tts")  # Default to edge-tts
        self.coqui_tts = None
        self.pyttsx3_engine = None
        self.edge_tts_tool = None
        self.current_model = None
        self.device = None
        self.temp_dir = tempfile.mkdtemp(prefix="talkie_tts_")
        
        # Track loaded models for different languages
        self.loaded_models = {}
        self.current_language = "en"
        
        # Track current audio process for stopping
        self.current_audio_process = None
        self.is_playing = False
        
        # Default to multilingual model if config says so
        self.use_multilingual = self.tts_config.get("use_multilingual", True)
        self.preferred_model = self.tts_config.get("coqui_model")
        
        # Initialize the preferred engine
        self._init_engine(self.current_engine)
    
    async def initialize(self):
        """Async initialization for web interface compatibility."""
        # Ensure engine is initialized
        if self.current_engine == "edge_tts" and self.edge_tts_tool is None:
            self._init_edge_tts()
        elif self.current_engine == "coqui" and self.coqui_tts is None:
            self._init_coqui_tts()
        elif self.current_engine == "pyttsx3" and self.pyttsx3_engine is None:
            self._init_pyttsx3()
        return self
    
    def _init_engine(self, engine: str):
        """Initialize the specified TTS engine."""
        print(f"🎯 Initializing TTS engine: {engine}")
        
        if engine == "edge_tts":
            self._init_edge_tts()
        elif engine == "coqui":
            self._init_coqui_tts()
        elif engine == "pyttsx3":
            self._init_pyttsx3()
        else:
            print(f"   ⚠️  Unknown engine '{engine}', trying edge_tts as default")
            self._init_edge_tts()
    
    def _init_edge_tts(self):
        """Initialize Edge TTS engine."""
        try:
            from tools.edge_tts_tool import EdgeTTSTool
            self.edge_tts_tool = EdgeTTSTool({"tts": self.tts_config})
            self.current_engine = "edge_tts"
            print("   ✅ Edge TTS ready")
            return True
        except Exception as e:
            print(f"   ⚠️  Edge TTS initialization failed: {e}")
            self.edge_tts_tool = None
            # Fall back to coqui
            self._init_coqui_tts()
            return False
    
    def get_available_engines(self) -> list:
        """Get list of available TTS engines."""
        return ["edge_tts", "coqui", "pyttsx3"]
    
    def get_current_engine(self) -> str:
        """Get current TTS engine name."""
        return self.current_engine
    
    def switch_engine(self, engine: str) -> dict:
        """Switch to a different TTS engine."""
        if engine not in self.get_available_engines():
            return {"success": False, "error": f"Unknown engine: {engine}"}
        
        print(f"🔄 Switching TTS engine from {self.current_engine} to {engine}")
        
        # Update config
        self.tts_config['engine'] = engine
        self.current_engine = engine
        
        # Initialize the new engine
        self._init_engine(engine)
        
        # Check if engine initialized successfully
        engine_ready = False
        if engine == "edge_tts" and self.edge_tts_tool is not None:
            engine_ready = True
        elif engine == "coqui" and self.coqui_tts is not None:
            engine_ready = True
        elif engine == "pyttsx3" and self.pyttsx3_engine is not None:
            engine_ready = True
        
        if engine_ready:
            return {"success": True, "engine": engine}
        else:
            # If the requested engine failed, we're now using a fallback
            actual_engine = self.get_current_engine()
            if actual_engine != engine:
                return {
                    "success": True, 
                    "engine": actual_engine,
                    "warning": f"{engine} not available, using {actual_engine} instead"
                }
            return {"success": False, "error": f"Failed to initialize {engine}"}
    
    def _detect_language(self, text: str) -> str:
        """Detect language from text."""
        # Check for Chinese characters
        if re.search(r'[\u4e00-\u9fff]', text):
            return "zh-cn"
        
        # Check for Japanese
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return "ja"
        
        # Check for Korean
        if re.search(r'[\uac00-\ud7af]', text):
            return "ko"
        
        # Check for Arabic
        if re.search(r'[\u0600-\u06ff]', text):
            return "ar"
        
        # Check for Cyrillic (Russian)
        if re.search(r'[\u0400-\u04ff]', text):
            return "ru"
        
        # Default to English
        return "en"
    
    def _preprocess_chinese_text(self, text: str) -> str:
        """
        Preprocess Chinese text for TTS.
        Removes or converts characters that the model doesn't support.
        """
        import re
        
        # Replace common English punctuation with Chinese equivalents
        replacements = {
            ',': '，',
            '.': '。',
            '?': '？',
            '!': '！',
            ':': '：',
            ';': '；',
            '"': '"',
            '"': '"',
            "'": ''',
            "'": ''',
            '(': '（',
            ')': '）',
            '[': '【',
            ']': '】',
            '{': '｛',
            '}': '｝',
        }
        
        for eng, chn in replacements.items():
            text = text.replace(eng, chn)
        
        # Remove English letters and numbers (the Chinese model doesn't handle them well)
        # Instead of removing, replace with their Chinese pronunciation
        text = re.sub(r'[a-zA-Z]+', '字母', text)  # Replace English words with "letters"
        text = re.sub(r'\d+', '数字', text)  # Replace numbers with "numbers"
        
        # Remove other special characters that might cause issues
        # Keep only Chinese characters, basic Chinese punctuation, and whitespace
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s]', '', text)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _get_model_for_language(self, language: str) -> str:
        """Get the appropriate model for the language."""
        language = language.lower()
        
        # If multilingual mode is enabled and language is supported by XTTS, use XTTS
        if self.use_multilingual and language in self.XTTS_LANGUAGES:
            return self.LANGUAGE_MODELS["multilingual"]
        
        # Otherwise use language-specific model if available
        if language in self.LANGUAGE_MODELS:
            return self.LANGUAGE_MODELS[language]
        
        # Fall back to multilingual model
        return self.LANGUAGE_MODELS["multilingual"]
    
    def _init_coqui_tts(self, language: str = "en", force_model: str = None):
        """Initialize Coqui TTS engine for a specific language."""
        try:
            import torch
            from TTS.api import TTS
            
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🎯 Initializing Coqui TTS on {self.device.upper()}")
            
            # Get the appropriate model
            if force_model:
                model = force_model
            elif self.preferred_model:
                # Check if preferred model supports this language
                model = self.preferred_model
            else:
                model = self._get_model_for_language(language)
            
            print(f"   Loading model: {model}")
            tts = TTS(model).to(self.device)
            
            self.coqui_tts = tts
            self.current_model = model
            self.current_language = language
            self.loaded_models[language] = tts
            
            print(f"   ✅ Coqui TTS ready with model: {model}")
            
            # Show speakers if available
            if hasattr(tts, 'speakers') and tts.speakers:
                print(f"   Available speakers: {len(tts.speakers)}")
                # Store speakers in config for web UI
                self.tts_config['speakers_list'] = tts.speakers
            
            return True
            
        except ImportError as e:
            print(f"   ⚠️  Coqui TTS not available: {e}")
            print("   Falling back to pyttsx3")
            self.coqui_tts = None
            # Try pyttsx3 as fallback
            self._init_pyttsx3()
            # Update current_engine to reflect fallback
            if self.pyttsx3_engine is not None:
                self.current_engine = "pyttsx3"
                self.tts_config['engine'] = "pyttsx3"
            return False
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"   ⚠️  CUDA out of memory, trying CPU...")
                try:
                    # Try loading on CPU instead
                    import torch
                    from TTS.api import TTS
                    self.device = "cpu"
                    # Get model again
                    if force_model:
                        model = force_model
                    elif self.preferred_model:
                        model = self.preferred_model
                    else:
                        model = self._get_model_for_language(language)
                    print(f"   Loading model on CPU: {model}")
                    tts = TTS(model).to(self.device)
                    
                    self.coqui_tts = tts
                    self.current_model = model
                    self.current_language = language
                    self.loaded_models[language] = tts
                    
                    print(f"   ✅ Coqui TTS ready on CPU with model: {model}")
                    
                    # Show speakers if available
                    if hasattr(tts, 'speakers') and tts.speakers:
                        print(f"   Available speakers: {len(tts.speakers)}")
                        # Store speakers in config for web UI
                        self.tts_config['speakers_list'] = tts.speakers
                    
                    return True
                except Exception as cpu_e:
                    print(f"   ⚠️  Failed to initialize Coqui TTS on CPU: {cpu_e}")
                    print("   Falling back to pyttsx3")
                    self.coqui_tts = None
                    return False
            else:
                print(f"   ⚠️  Failed to initialize Coqui TTS: {e}")
                print("   Falling back to pyttsx3")
                self.coqui_tts = None
                return False
        except Exception as e:
            print(f"   ⚠️  Failed to initialize Coqui TTS: {e}")
            print("   Falling back to pyttsx3")
            self.coqui_tts = None
            return False
    
    def _switch_model(self, language: str):
        """Switch to a different language model."""
        if language == self.current_language and self.coqui_tts:
            return True
        
        # Check if we're using XTTS-v2 (multilingual model)
        # XTTS-v2 supports all languages in one model, no need to switch
        if self.current_model and "xtts" in self.current_model.lower():
            self.current_language = language
            print(f"   🌐 Language: {language} (using XTTS-v2, no model switch needed)")
            return True
        
        # Check if model is already loaded
        if language in self.loaded_models:
            self.coqui_tts = self.loaded_models[language]
            self.current_language = language
            print(f"   🌐 Switched to {language} model")
            return True
        
        # Load new model - only if not using XTTS-v2
        print(f"   🌐 Loading {language} model...")
        
        # For non-XTTS models, use language-specific models
        if language in ["zh", "zh-cn"]:
            force_model = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
        elif language == "en":
            force_model = "tts_models/en/ljspeech/tacotron2-DDC"
        else:
            force_model = self._get_model_for_language(language)
        
        return self._init_coqui_tts(language, force_model=force_model)
    
    def _init_pyttsx3(self):
        """Initialize pyttsx3 as fallback."""
        try:
            import pyttsx3
            self.pyttsx3_engine = pyttsx3.init()
            self.pyttsx3_engine.setProperty('rate', self.tts_config.get('rate', 180))
            self.pyttsx3_engine.setProperty('volume', self.tts_config.get('volume', 1.0))
            print("   ✅ pyttsx3 fallback ready")
        except Exception as e:
            print(f"   ❌ pyttsx3 also failed: {e}")
            self.pyttsx3_engine = None
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration in seconds."""
        try:
            import wave
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            print(f"   [AUDIO DEBUG] Could not get audio duration: {e}")
            # Estimate based on file size (approx 176KB per second for 16-bit mono 44.1kHz)
            try:
                import os
                file_size = os.path.getsize(audio_path)
                estimated_duration = file_size / 176400
                return estimated_duration
            except:
                return 30  # Default fallback
    
    def _play_audio(self, audio_path: str) -> Optional[subprocess.Popen]:
        """Play audio file using available audio player - NON-BLOCKING.
        
        Returns:
            subprocess.Popen: The audio player process (can be terminated to stop)
            None: If no audio player found
        """
        import subprocess
        import time
        
        print(f" [AUDIO DEBUG] _play_audio() NON-BLOCKING called with: {audio_path}")

        # Try only ONE player - the first one that works
        players = [
            ("paplay", "PulseAudio"),  # First choice
            ("ffplay -autoexit -nodisp", "FFmpeg"),  # Second choice
        ]

        for player_cmd, player_name in players:
            player_executable = player_cmd.split()[0]
            try:
                result = subprocess.run(
                    f"which {player_executable}",
                    shell=True,
                    capture_output=True,
                    check=False
                )
                if result.returncode == 0:
                    cmd = f"{player_cmd} \"{audio_path}\""
                    print(f" [AUDIO DEBUG] Starting {player_name} (NON-BLOCKING)")
                    # Use Popen for non-blocking playback
                    process = subprocess.Popen(
                        cmd, 
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print(f" [AUDIO DEBUG] Started {player_name} with PID {process.pid}")
                    self.current_audio_process = process
                    self.is_playing = True
                    return process
            except Exception as e:
                print(f" [AUDIO DEBUG] {player_name} failed: {e}")
                continue

        # Fallback to aplay ONLY if others fail
        try:
            print(f" [AUDIO DEBUG] Trying aplay as fallback (NON-BLOCKING)")
            process = subprocess.Popen(
                f"aplay \"{audio_path}\"",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f" [AUDIO DEBUG] Started aplay with PID {process.pid}")
            self.current_audio_process = process
            self.is_playing = True
            return process
        except Exception as e:
            print(f" [AUDIO DEBUG] aplay also failed: {e}")

        print(f" [AUDIO DEBUG] No audio player found!")
        return None
    
    def stop_audio(self, reason: str = "general") -> bool:
        """Stop the current audio playback immediately.
        
        Args:
            reason: Why we're stopping - "chat" stops chat audio, "file" skips stopping for file reading
        """
        print(f" [AUDIO DEBUG] TTSTool stop_audio called. reason={reason}, current_process: {self.current_audio_process}, edge_tts_tool: {self.edge_tts_tool}")
        stopped = False
        
        # Stop local process (Coqui/pyttsx3)
        if self.current_audio_process and self.current_audio_process.poll() is None:
            try:
                print(f" [AUDIO DEBUG] Stopping local audio process {self.current_audio_process.pid}")
                self.current_audio_process.terminate()
                try:
                    self.current_audio_process.wait(timeout=1)
                except:
                    self.current_audio_process.kill()
                stopped = True
            except Exception as e:
                print(f" [AUDIO DEBUG] Error stopping local audio: {e}")
            self.is_playing = False
            self.current_audio_process = None
        
        # Delegate to edge_tts tool if it exists
        if self.edge_tts_tool and hasattr(self.edge_tts_tool, 'stop_audio'):
            try:
                print(f" [AUDIO DEBUG] Stopping EdgeTTS tool audio with reason={reason}")
                if self.edge_tts_tool.stop_audio(reason=reason):
                    stopped = True
            except Exception as e:
                print(f" [AUDIO DEBUG] Error stopping Edge TTS audio: {e}")
        
        if stopped:
            print(f" [AUDIO DEBUG] Audio stopped successfully")
            return True
        return False
    
    def wait_for_audio(self, timeout: float = None) -> bool:
        """Wait for current audio to finish playing.
        
        Args:
            timeout: Max time to wait in seconds (None = wait indefinitely)
            
        Returns:
            True if audio finished, False if timeout
        """
        if not self.current_audio_process:
            return True
            
        start_time = time.time()
        while self.current_audio_process.poll() is None:
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.05)  # Check every 50ms
        
        self.is_playing = False
        self.current_audio_process = None
        return True
    
    async def execute(self, text: str, language: str = "auto", speaker_id: str = None, 
                     speed: float = 1.0, audio_type: str = "chat") -> Dict[str, Any]:
        """Speak the given text using the selected TTS engine.
        
        Args:
            audio_type: "chat" for chat responses, "file" for file reading
        """
        
        if not text or not text.strip():
            return {
                "success": False,
                "error": "No text provided",
                "spoken": False
            }
        
        text = text.strip()
        TTSTool.call_count += 1
        import uuid
        call_id = str(uuid.uuid4())[:8]
        
        # Use configured speaker_id if none provided
        if speaker_id is None:
            speaker_id = self.tts_config.get('speaker_id')
        
        print(f"   [TTS] Executing TTS: {len(text)} chars, engine: {self.current_engine}, language: {language}")
        
        # Auto-detect language if not specified
        if language == "auto":
            language = self._detect_language(text)
            print(f"   🌐 Auto-detected language: {language}")
        
        # Show language indicator
        lang_indicator = {
            "en": "🇬🇧",
            "zh-cn": "🇨🇳",
            "zh": "🇨🇳",
            "es": "🇪🇸",
            "fr": "🇫🇷",
            "de": "🇩🇪",
            "ja": "🇯🇵",
            "ko": "🇰🇷",
            "ar": "🇸🇦",
            "ru": "🇷🇺",
        }.get(language.lower(), "🌐")
        
        short_text = text[:60] + "..." if len(text) > 60 else text
        print(f"🔊 Speaking {lang_indicator}: '{short_text}'")
        
        try:
            # Route to appropriate engine based on current_engine setting
            if self.current_engine == "edge_tts":
                if self.edge_tts_tool is None:
                    self._init_edge_tts()
                if self.edge_tts_tool:
                    return await self._speak_edge_tts(text, language, speed, audio_type)
                # Fallback to coqui if edge-tts fails
                print("   Edge TTS not available, falling back to Coqui")
                
            if self.current_engine == "coqui" or (self.current_engine == "edge_tts" and self.edge_tts_tool is None):
                # Use Coqui TTS if available
                if self.coqui_tts is not None or self._init_coqui_tts(language):
                    # Switch model if needed
                    if language != self.current_language:
                        if not self._switch_model(language):
                            print(f"   ⚠️  Could not load {language} model, using current model")
                    
                    return await self._speak_coqui(text, language, speaker_id, speed)
            
            # Fall back to pyttsx3
            if self.pyttsx3_engine is not None or self._init_pyttsx3():
                return await self._speak_pyttsx3(text, language)
            
            return {
                "success": False,
                "error": "No TTS engine available",
                "spoken": False
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "spoken": False
            }
    
    async def _speak_coqui(self, text: str, language: str, speaker_id: str = None, 
                          speed: float = 1.0) -> Dict[str, Any]:
        """Speak using Coqui TTS."""
        import asyncio
        
        # Preprocess text for Chinese model
        if language in ["zh", "zh-cn"] and "zh-CN" in self.current_model:
            original_text = text
            text = self._preprocess_chinese_text(text)
            if text != original_text:
                print(f"   📝 Preprocessed Chinese text")
        
        # Generate temporary audio file
        audio_file = os.path.join(self.temp_dir, f"tts_{language}_{hash(text)}.wav")
        
        try:
            # Run TTS in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Check if using XTTS (multilingual model)
            is_xtts = "xtts" in self.current_model.lower()
            
            import sys
            from io import StringIO
            
            # Suppress stderr to hide "Character X not found" warnings
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            
            try:
                if is_xtts:
                    # XTTS-v2 supports multiple languages and speakers
                    await loop.run_in_executor(
                        None,
                        lambda: self.coqui_tts.tts_to_file(
                            text=text,
                            speaker=speaker_id or self.coqui_tts.speakers[0] if hasattr(self.coqui_tts, 'speakers') and self.coqui_tts.speakers else None,
                            language=language if language in self.XTTS_LANGUAGES else "en",
                            file_path=audio_file
                        )
                    )
                else:
                    # Single language model
                    await loop.run_in_executor(
                        None,
                        lambda: self.coqui_tts.tts_to_file(
                            text=text,
                            file_path=audio_file
                        )
                    )
            finally:
                # Restore stderr
                sys.stderr = old_stderr

            # Play the audio (NON-BLOCKING - returns process)
            audio_process = None
            if os.path.exists(audio_file):
                audio_process = self._play_audio(audio_file)
                return {
                    "success": True,
                    "spoken": True,
                    "text": text,
                    "language": language,
                    "characters": len(text),
                    "engine": "coqui-tts",
                    "model": self.current_model,
                    "multilingual": is_xtts,
                    "audio_process": audio_process,
                    "audio_file": audio_file  # Return path for cleanup later
                }
            else:
                return {
                    "success": False,
                    "error": "Audio file was not generated",
                    "spoken": False
                }
                
        except Exception as e:
            # Clean up on error
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass
            
            # Try fallback to pyttsx3
            if self.pyttsx3_engine:
                print(f"   Coqui TTS failed, falling back to pyttsx3: {e}")
                return await self._speak_pyttsx3(text, language)
            
            raise e
    
    async def _speak_pyttsx3(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Speak using pyttsx3 fallback."""
        import asyncio
        
        loop = asyncio.get_event_loop()
        
        def speak():
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
        
        await loop.run_in_executor(None, speak)
        
        return {
            "success": True,
            "spoken": True,
            "text": text,
            "language": language,
            "characters": len(text),
            "engine": "pyttsx3",
            "note": "Using fallback TTS engine (limited language support)"
        }
    
    async def _speak_edge_tts(self, text: str, language: str = "en", speed: float = 1.0, audio_type: str = "chat") -> Dict[str, Any]:
        """Speak using Microsoft Edge TTS."""
        if self.edge_tts_tool is None:
            return {
                "success": False,
                "error": "Edge TTS not initialized",
                "spoken": False
            }
        
        # Set audio type before executing
        if hasattr(self.edge_tts_tool, 'set_audio_type'):
            self.edge_tts_tool.set_audio_type(audio_type)
        
        # Get voice based on detected language (not user's configured voice)
        # This allows auto-switching to Chinese voice for Chinese content
        voice = self.edge_tts_tool._get_voice_for_language(language)
        
        # Call Edge TTS with the appropriate voice for the language
        result = await self.edge_tts_tool.execute(text=text, voice=voice, speed=speed)
        
        # Add engine info
        if result.get("success"):
            result["engine"] = "edge-tts"
            result["language"] = language
            result["voice"] = voice
        
        return result
    
    def get_available_voices(self) -> Dict[str, Any]:
        """Get list of available voices/speakers."""
        voices = []
        
        if self.coqui_tts:
            try:
                if hasattr(self.coqui_tts, 'speakers') and self.coqui_tts.speakers:
                    voices = self.coqui_tts.speakers
                else:
                    voices = ["default"]
            except:
                voices = ["default"]
        
        elif self.pyttsx3_engine:
            try:
                voices = [v.id for v in self.pyttsx3_engine.getProperty('voices')]
            except:
                voices = ["default"]
        
        return {
            "voices": voices,
            "engine": "coqui-tts" if self.coqui_tts else "pyttsx3",
            "model": self.current_model if self.coqui_tts else None,
            "current_language": self.current_language,
            "supported_languages": self.XTTS_LANGUAGES if self.coqui_tts and "xtts" in self.current_model.lower() else ["en"]
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages."""
        return {
            "en": "English",
            "zh": "Chinese (中文)",
            "zh-cn": "Chinese Simplified (简体中文)",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "pl": "Polish",
            "tr": "Turkish",
            "ru": "Russian",
            "nl": "Dutch",
            "cs": "Czech",
            "ar": "Arabic",
            "ja": "Japanese",
            "hu": "Hungarian",
            "ko": "Korean",
            "hi": "Hindi",
        }
    
    def get_available_speakers(self) -> list:
        """Get list of available speakers/personas for the current model."""
        speakers = []
        
        # Return Edge TTS voices if using Edge TTS
        if self.current_engine == "edge_tts":
            if self.edge_tts_tool is None:
                self._init_edge_tts()
            if self.edge_tts_tool:
                voices = self.edge_tts_tool.get_available_voices()
                for voice in voices:
                    speakers.append({
                        "id": voice["id"],
                        "name": voice["name"],
                        "gender": voice["gender"],
                        "locale": voice["locale"],
                        "type": "edge"
                    })
            return speakers
        
        # For Coqui TTS, only initialize if it's the current engine
        if self.current_engine == "coqui":
            if self.coqui_tts is None:
                print("🔄 Coqui TTS not initialized, attempting to initialize...")
                self._init_coqui_tts()
            
            if self.coqui_tts:
                # Check if XTTS model with speakers
                if hasattr(self.coqui_tts, 'speakers') and self.coqui_tts.speakers:
                    for idx, speaker in enumerate(self.coqui_tts.speakers):
                        speakers.append({
                            "id": speaker,
                            "name": speaker,
                            "index": idx,
                            "type": "xtts"
                        })
                elif "xtts" in self.current_model.lower():
                    # XTTS model but speakers not loaded yet, return default
                    speakers = [{"id": "default", "name": "Default Voice", "index": 0, "type": "xtts"}]
        
        # For pyttsx3
        if self.current_engine == "pyttsx3":
            if self.pyttsx3_engine is None:
                self._init_pyttsx3()
            if self.pyttsx3_engine:
                try:
                    for idx, voice in enumerate(self.pyttsx3_engine.getProperty('voices')):
                        speakers.append({
                            "id": voice.id,
                            "name": voice.name,
                            "index": idx,
                            "type": "pyttsx3"
                        })
                except:
                    pass
        
        return speakers
    
    def set_speaker(self, speaker_id: str) -> bool:
        """Set the current speaker/persona."""
        try:
            # Handle Edge TTS voice selection
            if self.current_engine == "edge_tts":
                if self.edge_tts_tool is None:
                    self._init_edge_tts()
                if self.edge_tts_tool:
                    # Always set the voice, even if same as current (to force re-init)
                    self.edge_tts_tool.set_voice(speaker_id)
                    self.edge_tts_tool.current_voice = speaker_id
                    self.tts_config['edge_voice'] = speaker_id
                    print(f"   🎭 Edge TTS voice set to: {speaker_id}")
                    return True
                return False
            
            # Handle Coqui TTS speaker selection
            if self.current_engine == "coqui":
                if self.coqui_tts and hasattr(self.coqui_tts, 'speakers'):
                    if speaker_id in self.coqui_tts.speakers:
                        self.tts_config['speaker_id'] = speaker_id
                        print(f"   🎭 TTS speaker set to: {speaker_id}")
                        return True
                    else:
                        print(f"   ⚠️  Speaker '{speaker_id}' not found in available speakers")
                        return False
            
            # Handle pyttsx3 voice selection
            if self.current_engine == "pyttsx3" and self.pyttsx3_engine:
                self.pyttsx3_engine.setProperty('voice', speaker_id)
                self.tts_config['speaker_id'] = speaker_id
                return True
            
            return False
        except Exception as e:
            print(f"   ⚠️  Failed to set speaker: {e}")
            return False
    
    def get_current_speaker(self) -> str:
        """Get the current speaker ID."""
        # For Edge TTS, return the current voice
        if self.current_engine == "edge_tts":
            if self.edge_tts_tool:
                return self.edge_tts_tool.get_current_voice()
            return self.tts_config.get('edge_voice', 'en-US-AriaNeural')
        
        # For Coqui TTS
        speaker = self.tts_config.get('speaker_id')
        if speaker is None:
            # Return first available speaker if XTTS, otherwise 'default'
            if self.coqui_tts and hasattr(self.coqui_tts, 'speakers') and self.coqui_tts.speakers:
                return self.coqui_tts.speakers[0]
            return 'default'
        return speaker
    
    def __del__(self):
        """Cleanup temporary files."""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
