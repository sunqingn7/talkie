"""
Qwen3 TTS Tool - Local Text-to-Speech using Qwen3-TTS models
Supports CustomVoice (predefined speakers) and VoiceDesign (natural language voice design).
Can use streaming TTS server for lower latency.
"""

import asyncio
import os
import tempfile
import subprocess
from typing import Any, Dict, List, Optional
from pathlib import Path
import time

from tools import BaseTool


class QwenTTSTool(BaseTool):
    """Tool to convert text to speech using Qwen3-TTS models.

    Features:
    - Local inference (no internet required after model download)
    - Two model types:
      - CustomVoice: Predefined speakers with instruction control
      - VoiceDesign: Natural language voice design
    - Supports 10 languages
    - Streaming support for low latency
    """

    CUSTOM_VOICE_SPEAKERS = [
        {
            "id": "Vivian",
            "name": "Vivian",
            "description": "Bright, slightly edgy young female voice",
            "native_language": "Chinese",
        },
        {
            "id": "Serena",
            "name": "Serena",
            "description": "Warm, gentle young female voice",
            "native_language": "Chinese",
        },
        {
            "id": "Uncle_Fu",
            "name": "Uncle Fu",
            "description": "Seasoned male voice with a low, mellow timbre",
            "native_language": "Chinese",
        },
        {
            "id": "Dylan",
            "name": "Dylan",
            "description": "Youthful Beijing male voice (Beijing Dialect)",
            "native_language": "Chinese",
        },
        {
            "id": "Eric",
            "name": "Eric",
            "description": "Lively Chengdu male voice (Sichuan Dialect)",
            "native_language": "Chinese",
        },
        {
            "id": "Ryan",
            "name": "Ryan",
            "description": "Dynamic male voice with strong rhythmic drive",
            "native_language": "English",
        },
        {
            "id": "Aiden",
            "name": "Aiden",
            "description": "Sunny American male voice with a clear midrange",
            "native_language": "English",
        },
        {
            "id": "Ono_Anna",
            "name": "Ono Anna",
            "description": "Playful Japanese female voice",
            "native_language": "Japanese",
        },
        {
            "id": "Sohee",
            "name": "Sohee",
            "description": "Warm Korean female voice with rich emotion",
            "native_language": "Korean",
        },
    ]

    SUPPORTED_LANGUAGES = [
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
        "Auto",
    ]

    MODEL_TYPES = {
        "custom_voice_0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "custom_voice_1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "voice_design_1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }

    def _get_description(self) -> str:
        return (
            "Convert text to speech using local Qwen3-TTS models. "
            "Supports predefined speakers (CustomVoice) or natural language voice design (VoiceDesign). "
            "Runs locally without internet after model download."
        )

    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to speak aloud"},
                "model_type": {
                    "type": "string",
                    "enum": [
                        "custom_voice_0.6b",
                        "custom_voice_1.7b",
                        "voice_design_1.7b",
                    ],
                    "description": "Model type to use",
                    "default": "custom_voice_0.6b",
                },
                "speaker": {
                    "type": "string",
                    "description": "Speaker ID for CustomVoice models (e.g., 'Vivian', 'Ryan')",
                    "default": "Vivian",
                },
                "language": {
                    "type": "string",
                    "description": "Language code (Chinese, English, Japanese, etc.) or 'Auto' for auto-detect",
                    "default": "Auto",
                },
                "instruct": {
                    "type": "string",
                    "description": "Natural language instruction for voice control (optional)",
                    "default": "",
                },
            },
            "required": ["text"],
        }

    def __init__(self, config: dict):
        super().__init__(config)
        self.tts_config = config.get("tts", {})
        qwen_config = self.tts_config.get("qwen_tts", {})

        self.temp_dir = tempfile.mkdtemp(prefix="talkie_qwen_tts_")
        self.model_type = qwen_config.get("model_type", "custom_voice_0.6b")
        self.current_speaker = qwen_config.get("speaker", "Vivian")
        self.current_language = qwen_config.get("language", "Auto")
        self.current_instruct = qwen_config.get("instruct", "")
        self.voice_output = self.tts_config.get("voice_output", "local")
        self.current_audio_process = None
        self.is_playing = False
        self.current_audio_type = None
        self._model = None
        self._model_loaded = False
        self._model_loading = False
        self._load_error = None
        self._device = qwen_config.get("device", "cuda:0")
        self._dtype = qwen_config.get("dtype", "bfloat16")

        self.use_streaming_server = qwen_config.get("use_streaming_server", True)
        self.streaming_server_host = qwen_config.get(
            "streaming_server_host", "localhost"
        )
        self.streaming_server_port = qwen_config.get("streaming_server_port", 8083)
        self._server_client = None
        self._server_start_attempted = False

        # Always load local model as backup (in case streaming server fails)
        # This ensures we have a working TTS even if server crashes
        self._start_background_load()

    def _get_server_client(self, auto_start: bool = False):
        """Get or create TTS server client."""
        if self._server_client is None:
            try:
                from tools.tts_server_client import TTSServerClient

                self._server_client = TTSServerClient(
                    host=self.streaming_server_host,
                    port=self.streaming_server_port,
                    auto_start=False,  # Don't auto-start, we control it manually
                )
            except ImportError:
                print("[Qwen TTS] TTS server client not available")
                return None
        return self._server_client

    def is_server_available(self) -> bool:
        """Check if TTS streaming server is available."""
        client = self._get_server_client()
        if client:
            return client.is_server_running()
        return False

    def start_streaming_server(self) -> bool:
        """Start the TTS streaming server."""
        if self._server_start_attempted:
            # Already tried to start, just check if it's running
            client = self._get_server_client()
            if client and client.is_server_running():
                return True
            return False

        self._server_start_attempted = True
        client = self._get_server_client()
        if client:
            return client.start_server()
        return False

    def set_audio_type(self, audio_type: str):
        """Set the type of audio that will be played (chat or file)."""
        self.current_audio_type = audio_type
        print(f"[Qwen TTS] Set audio_type to {audio_type}")

    def get_available_speakers(self) -> List[Dict]:
        """Get list of available speakers for CustomVoice models."""
        return self.CUSTOM_VOICE_SPEAKERS

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.SUPPORTED_LANGUAGES

    def get_model_types(self) -> List[Dict]:
        """Get list of available model types."""
        return [
            {
                "id": "custom_voice_0.6b",
                "name": "Qwen3-TTS 0.6B CustomVoice",
                "description": "Faster, smaller model with predefined speakers",
            },
            {
                "id": "custom_voice_1.7b",
                "name": "Qwen3-TTS 1.7B CustomVoice",
                "description": "Higher quality model with predefined speakers",
            },
            {
                "id": "voice_design_1.7b",
                "name": "Qwen3-TTS 1.7B VoiceDesign",
                "description": "Design voices using natural language descriptions",
            },
        ]

    def set_model_type(self, model_type: str):
        """Set the model type and reset loaded model."""
        if model_type in self.MODEL_TYPES:
            self.model_type = model_type
            self._model_loaded = False
            self._model = None
            print(f"[Qwen TTS] Model type set to: {model_type}")

    def set_speaker(self, speaker: str):
        """Set the speaker for CustomVoice models."""
        valid_speakers = [s["id"] for s in self.CUSTOM_VOICE_SPEAKERS]
        if speaker in valid_speakers:
            self.current_speaker = speaker
            print(f"[Qwen TTS] Speaker set to: {speaker}")
        else:
            print(
                f"[Qwen TTS] Warning: Invalid speaker '{speaker}', keeping '{self.current_speaker}'"
            )

    def set_language(self, language: str):
        """Set the language."""
        if (
            language in self.SUPPORTED_LANGUAGES
            or language.capitalize() in self.SUPPORTED_LANGUAGES
        ):
            self.current_language = (
                language.capitalize()
                if language.capitalize() in self.SUPPORTED_LANGUAGES
                else language
            )
            print(f"[Qwen TTS] Language set to: {self.current_language}")

    def set_instruct(self, instruct: str):
        """Set the instruction for voice control."""
        self.current_instruct = instruct
        print(
            f"[Qwen TTS] Instruct set to: {instruct[:50]}..."
            if len(instruct) > 50
            else f"[Qwen TTS] Instruct set to: {instruct}"
        )

    def _start_background_load(self):
        """Start loading model in background thread."""
        import threading
        import time

        def load_in_background():
            # Small delay to let server fully start
            time.sleep(2)
            self._model_loading = True
            self._load_error = None
            try:
                self._load_model_sync()
            except Exception as e:
                self._load_error = str(e)
                print(f"[Qwen TTS] Background load failed: {e}")
            finally:
                self._model_loading = False

        thread = threading.Thread(target=load_in_background, daemon=True)
        thread.start()
        print("[Qwen TTS] Scheduled model loading in background (2s delay)...")

    def _load_model_sync(self):
        """Synchronously load the model using qwen-tts library."""
        import time
        import torch
        from qwen_tts import Qwen3TTSModel

        model_id = self.MODEL_TYPES.get(self.model_type)
        if not model_id:
            raise ValueError(f"Unknown model type: {self.model_type}")

        print(f"[Qwen TTS] Loading model: {model_id}")
        load_start = time.time()

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self._dtype, torch.bfloat16)

        # Load model using qwen-tts library
        self._model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=self._device,
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )

        self._model_loaded = True
        load_time = time.time() - load_start
        print(f"[Qwen TTS] Model loaded successfully in {load_time:.2f}s!")

        # Warm up the model in background
        import threading

        threading.Thread(target=self._do_warmup, daemon=True).start()

    def _load_model(self):
        """Load the Qwen3-TTS model (waits if loading in background)."""
        if self._model_loaded and self._model is not None:
            return True

        if self._load_error:
            print(f"[Qwen TTS] Model load previously failed: {self._load_error}")
            return False

        # If model is loading in background, wait for it
        if self._model_loading:
            print(f"[Qwen TTS] Waiting for background model load to complete...")
            import time

            max_wait = 120  # 2 minutes
            waited = 0
            while self._model_loading and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5

            if self._model_loaded:
                return True
            if self._load_error:
                print(f"[Qwen TTS] Model load failed: {self._load_error}")
                return False
            print(f"[Qwen TTS] Model load timed out")
            return False

        # Not loading, not loaded - start loading
        self._start_background_load()
        return self._load_model()  # Recursively wait

    def _play_audio(self, audio_file: str):
        """Play audio file using system player."""
        try:
            if os.path.exists(audio_file):
                cmd = [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    audio_file,
                ]
                proc = subprocess.Popen(cmd)
                return proc
        except Exception as e:
            print(f"[Qwen TTS] Error playing audio: {e}")
        return None

    def stop_audio(self, reason: str = ""):
        """Stop current audio playback."""
        # If we're asked to stop for chat but current audio is file reading, don't stop
        if reason == "chat" and self.current_audio_type == "file":
            print(f"[Qwen TTS] Skipping stop - current audio is file reading")
            return

        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
                self.current_audio_process.wait(timeout=0.5)
            except:
                try:
                    self.current_audio_process.kill()
                except:
                    pass
            self.current_audio_process = None
        self.is_playing = False
        if reason:
            print(f"[Qwen TTS] Audio stopped: {reason}")

    def _do_warmup(self):
        """Internal warmup - generates a short audio to warm up the model."""
        try:
            import asyncio
            import time

            time.sleep(0.2)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.execute(text="测试", language="zh-cn", audio_type="chat")
                )
                if result.get("success"):
                    print("[Qwen TTS] Warmup completed successfully")
                else:
                    print(f"[Qwen TTS] Warmup failed: {result.get('error')}")
            finally:
                loop.close()
        except Exception as e:
            print(f"[Qwen TTS] Warmup error: {e}")

    async def execute(
        self,
        text: str,
        model_type: str = None,
        speaker: str = None,
        language: str = None,
        instruct: str = None,
        audio_type: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute TTS conversion."""

        if not text or not text.strip():
            return {"success": False, "error": "No text provided"}

        if model_type:
            self.set_model_type(model_type)
        if speaker:
            self.set_speaker(speaker)
        if language:
            self.set_language(language)
        if instruct is not None:
            self.set_instruct(instruct)
        if audio_type:
            self.set_audio_type(audio_type)

        if self.use_streaming_server:
            result = await self._execute_via_server(text)
            if result.get("success"):
                return result
            # Server not ready, fall back to local
            print(
                f"[Qwen TTS] Streaming server not ready, using local: {result.get('error')}"
            )

        return await self._execute_local(text)

    async def _execute_via_server(self, text: str) -> Dict[str, Any]:
        """Execute TTS via streaming server."""
        client = self._get_server_client()
        if client is None:
            return {"success": False, "error": "TTS server client not available"}

        # Check if server is running and ready
        if not client.is_server_running():
            if not self._server_start_attempted:
                # First time - try to start the server
                print("[Qwen TTS] Starting TTS streaming server...")
                self.start_streaming_server()

            # After starting (or if already attempted), check if ready
            if not client.is_server_running():
                # Server process started but not ready yet (loading model)
                return {
                    "success": False,
                    "error": "TTS server not ready (still loading model)",
                }

        try:
            print(f"[Qwen TTS] Generating speech via streaming server: {text[:50]}...")
            start_time = time.time()

            audio_chunks = []
            async for chunk in client.synthesize_stream(
                text=text,
                backend="qwen_tts",
                voice=self.current_speaker,
                language=self._normalize_language_for_server(self.current_language),
                speed=1.0,
            ):
                audio_chunks.append(chunk)

            if not audio_chunks:
                return {"success": False, "error": "No audio received from server"}

            audio_data = b"".join(audio_chunks)
            gen_time = time.time() - start_time
            print(f"[Qwen TTS] Server generated in {gen_time:.2f}s")

            output_file = os.path.join(
                self.temp_dir, f"qwen_tts_{int(time.time())}.wav"
            )
            with open(output_file, "wb") as f:
                f.write(audio_data)

            audio_process = None
            if self.voice_output != "web":
                audio_process = self._play_audio(output_file)
                if audio_process:
                    self.current_audio_process = audio_process
                    self.is_playing = True

            return {
                "success": True,
                "text": text,
                "model_type": self.model_type,
                "speaker": self.current_speaker,
                "language": self.current_language,
                "output_file": output_file,
                "audio_file": output_file,
                "audio_process": audio_process,
                "voice_output": self.voice_output,
                "generation_time": gen_time,
                "via_server": True,
            }

        except Exception as e:
            print(f"[Qwen TTS] Server synthesis error: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": f"Server synthesis failed: {str(e)}"}

    async def _execute_local(self, text: str) -> Dict[str, Any]:
        """Execute TTS locally using qwen-tts library with pyaudio streaming."""

        if not self._load_model():
            return {
                "success": False,
                "error": "Failed to load Qwen3-TTS model. Make sure qwen-tts is installed and model is downloaded.",
            }

        try:
            import soundfile as sf
            import numpy as np
            import pyaudio

            print(f"[Qwen TTS] Generating speech for: {text[:50]}...")

            is_voice_design = "voice_design" in self.model_type
            lang = self.current_language if self.current_language != "Auto" else "Auto"

            start_time = time.time()

            print(
                f"[Qwen TTS] About to call model.generate, text_len={len(text)}, is_voice_design={is_voice_design}"
            )

            if is_voice_design:
                wavs, sr = self._model.generate_voice_design(
                    text=text,
                    language=lang,
                    instruct=self.current_instruct,
                )
            else:
                wavs, sr = self._model.generate_custom_voice(
                    text=text,
                    language=lang,
                    speaker=self.current_speaker,
                    instruct=self.current_instruct,
                )

            gen_time = time.time() - start_time
            print(
                f"[Qwen TTS] Generated in {gen_time:.2f}s, wav shape: {wavs[0].shape if hasattr(wavs[0], 'shape') else 'N/A'}"
            )

            output_file = os.path.join(
                self.temp_dir, f"qwen_tts_{int(time.time())}.wav"
            )
            sf.write(output_file, wavs[0], sr)

            audio_process = None

            # Stream to pyaudio if local playback
            if self.voice_output != "web":
                try:
                    # Convert to the format needed for pyaudio
                    audio_np = np.array(wavs[0], dtype=np.float32)

                    # Apply Hann window for anti-click
                    if len(audio_np) > 0:
                        window = np.hanning(len(audio_np))
                        audio_np = audio_np * window

                    # Convert to int16 for pyaudio
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32768.0

                    p = pyaudio.PyAudio()
                    stream = p.open(
                        format=pyaudio.paFloat32,
                        channels=1,
                        rate=sr,
                        output=True,
                        frames_per_buffer=1024,
                    )

                    # Write directly to stream
                    stream.write(audio_float.tobytes())

                    # Clean up
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    print(f"[Qwen TTS] Streamed to pyaudio")
                except Exception as e:
                    print(
                        f"[Qwen TTS] pyaudio stream error: {e}, falling back to file playback"
                    )
                    audio_process = self._play_audio(output_file)
            else:
                print(f"[Qwen TTS] voice_output=web, skipping local playback")

            if audio_process:
                self.current_audio_process = audio_process
                self.is_playing = True

            return {
                "success": True,
                "text": text,
                "model_type": self.model_type,
                "speaker": self.current_speaker if not is_voice_design else None,
                "language": self.current_language,
                "instruct": self.current_instruct,
                "output_file": output_file,
                "audio_file": output_file,
                "audio_process": audio_process,
                "voice_output": self.voice_output,
                "generation_time": gen_time,
            }

        except Exception as e:
            print(f"[Qwen TTS] Error during generation: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": f"Qwen TTS failed: {str(e)}"}

        try:
            import soundfile as sf
            import numpy as np
            import torch
            import pyaudio

            print(f"[Qwen TTS] Generating speech for: {text[:50]}...")

            is_voice_design = "voice_design" in self.model_type

            start_time = time.time()

            print(
                f"[Qwen TTS] About to call model.generate, text_len={len(text)}, is_voice_design={is_voice_design}"
            )

            # Tokenize using 12Hz tokenizer
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
            ).to(self._model.device)

            with torch.no_grad():
                # Generate codes autoregressively
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                )

                # Decode to waveform
                audio = self._model.decode(generated_ids)

            # Convert to numpy
            audio_np = audio.cpu().numpy().squeeze()

            # Apply Hann window for seamless stitching (anti-click)
            if len(audio_np) > 0:
                window = np.hanning(len(audio_np))
                audio_np = audio_np * window

            gen_time = time.time() - start_time
            print(
                f"[Qwen TTS] Generated in {gen_time:.2f}s, wav shape: {audio_np.shape}"
            )

            output_file = os.path.join(
                self.temp_dir, f"qwen_tts_{int(time.time())}.wav"
            )

            # Convert to float32 for soundfile
            audio_float32 = audio_np.astype(np.float32)
            sf.write(output_file, audio_float32, 24000)  # Qwen3 uses 24kHz

            audio_process = None

            # Stream to pyaudio if local playback
            if self.voice_output != "web":
                try:
                    p = pyaudio.PyAudio()
                    stream = p.open(
                        format=pyaudio.paFloat32,
                        channels=1,
                        rate=24000,
                        output=True,
                        frames_per_buffer=1024,
                    )

                    # Write audio directly to stream for real-time playback
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32768.0
                    stream.write(audio_float.tobytes())

                    # Clean up pyaudio
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    print(f"[Qwen TTS] Streamed to pyaudio")
                except Exception as e:
                    print(
                        f"[Qwen TTS] pyaudio stream error: {e}, falling back to file playback"
                    )
                    audio_process = self._play_audio(output_file)
            else:
                print(f"[Qwen TTS] voice_output=web, skipping local playback")

            if audio_process:
                self.current_audio_process = audio_process
                self.is_playing = True

            return {
                "success": True,
                "text": text,
                "model_type": self.model_type,
                "speaker": self.current_speaker if not is_voice_design else None,
                "language": self.current_language,
                "instruct": self.current_instruct,
                "output_file": output_file,
                "audio_file": output_file,
                "audio_process": audio_process,
                "voice_output": self.voice_output,
                "generation_time": gen_time,
            }

        except Exception as e:
            print(f"[Qwen TTS] Error during generation: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": f"Qwen TTS failed: {str(e)}"}

        try:
            import soundfile as sf
            import numpy as np
            import torch

            print(f"[Qwen TTS] Generating speech for: {text[:50]}...")

            is_voice_design = "voice_design" in self.model_type

            start_time = time.time()

            print(
                f"[Qwen TTS] About to call model.generate, text_len={len(text)}, is_voice_design={is_voice_design}"
            )

            # Tokenize using 12Hz tokenizer
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
            ).to(self._model.device)

            with torch.no_grad():
                # Generate codes autoregressively
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                )

                # Decode to waveform
                audio = self._model.decode(generated_ids)

            # Convert to numpy
            audio_np = audio.cpu().numpy().squeeze()

            # Apply Hann window for seamless stitching (anti-click)
            if len(audio_np) > 0:
                window = np.hanning(len(audio_np))
                audio_np = audio_np * window

            gen_time = time.time() - start_time
            print(
                f"[Qwen TTS] Generated in {gen_time:.2f}s, wav shape: {audio_np.shape}"
            )

            output_file = os.path.join(
                self.temp_dir, f"qwen_tts_{int(time.time())}.wav"
            )

            # Convert to float32 for soundfile
            audio_float32 = audio_np.astype(np.float32)
            sf.write(output_file, audio_float32, 24000)  # Qwen3 uses 24kHz

            audio_process = None

            if self.voice_output != "web":
                time.sleep(0.1)
                audio_process = self._play_audio(output_file)
                if audio_process:
                    self.current_audio_process = audio_process
                    self.is_playing = True
            else:
                print(f"[Qwen TTS] voice_output=web, skipping local playback")

            return {
                "success": True,
                "text": text,
                "model_type": self.model_type,
                "speaker": self.current_speaker if not is_voice_design else None,
                "language": self.current_language,
                "instruct": self.current_instruct,
                "output_file": output_file,
                "audio_file": output_file,
                "audio_process": audio_process,
                "voice_output": self.voice_output,
                "generation_time": gen_time,
            }

        except Exception as e:
            print(f"[Qwen TTS] Error during generation: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": f"Qwen TTS failed: {str(e)}"}

        try:
            import soundfile as sf

            print(f"[Qwen TTS] Generating speech for: {text[:50]}...")

            is_voice_design = "voice_design" in self.model_type
            lang = self.current_language if self.current_language != "Auto" else "Auto"

            start_time = time.time()

            print(
                f"[Qwen TTS] About to call model.generate, text_len={len(text)}, is_voice_design={is_voice_design}"
            )

            if is_voice_design:
                wavs, sr = self._model.generate_voice_design(
                    text=text,
                    language=lang,
                    instruct=self.current_instruct,
                )
            else:
                wavs, sr = self._model.generate_custom_voice(
                    text=text,
                    language=lang,
                    speaker=self.current_speaker,
                    instruct=self.current_instruct,
                )

            gen_time = time.time() - start_time
            print(
                f"[Qwen TTS] Generated in {gen_time:.2f}s, wav shape: {wavs[0].shape if hasattr(wavs[0], 'shape') else 'N/A'}"
            )

            output_file = os.path.join(
                self.temp_dir, f"qwen_tts_{int(time.time())}.wav"
            )
            sf.write(output_file, wavs[0], sr)

            audio_process = None

            if self.voice_output != "web":
                time.sleep(0.1)
                audio_process = self._play_audio(output_file)
                if audio_process:
                    self.current_audio_process = audio_process
                    self.is_playing = True
            else:
                print(f"[Qwen TTS] voice_output=web, skipping local playback")

            return {
                "success": True,
                "text": text,
                "model_type": self.model_type,
                "speaker": self.current_speaker if not is_voice_design else None,
                "language": self.current_language,
                "instruct": self.current_instruct,
                "output_file": output_file,
                "audio_file": output_file,
                "audio_process": audio_process,
                "voice_output": self.voice_output,
                "generation_time": gen_time,
            }

        except Exception as e:
            print(f"[Qwen TTS] Error during generation: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": f"Qwen TTS failed: {str(e)}"}

    def _normalize_language_for_server(self, language: str) -> str:
        """Normalize language for the TTS server."""
        if not language or language == "Auto":
            return "Auto"

        lang_map = {
            "Chinese": "zh",
            "English": "en",
            "Japanese": "ja",
            "Korean": "ko",
            "German": "de",
            "French": "fr",
            "Russian": "ru",
            "Portuguese": "pt",
            "Spanish": "es",
            "Italian": "it",
        }

        return lang_map.get(language, language.lower()[:2])

    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil

            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"[Qwen TTS] Error cleaning up: {e}")
