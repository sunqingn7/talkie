"""
Qwen3 TTS Tool - Local Text-to-Speech using Qwen3-TTS models
Supports CustomVoice (predefined speakers) and VoiceDesign (natural language voice design).
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
        {"id": "Vivian", "name": "Vivian", "description": "Bright, slightly edgy young female voice", "native_language": "Chinese"},
        {"id": "Serena", "name": "Serena", "description": "Warm, gentle young female voice", "native_language": "Chinese"},
        {"id": "Uncle_Fu", "name": "Uncle Fu", "description": "Seasoned male voice with a low, mellow timbre", "native_language": "Chinese"},
        {"id": "Dylan", "name": "Dylan", "description": "Youthful Beijing male voice (Beijing Dialect)", "native_language": "Chinese"},
        {"id": "Eric", "name": "Eric", "description": "Lively Chengdu male voice (Sichuan Dialect)", "native_language": "Chinese"},
        {"id": "Ryan", "name": "Ryan", "description": "Dynamic male voice with strong rhythmic drive", "native_language": "English"},
        {"id": "Aiden", "name": "Aiden", "description": "Sunny American male voice with a clear midrange", "native_language": "English"},
        {"id": "Ono_Anna", "name": "Ono Anna", "description": "Playful Japanese female voice", "native_language": "Japanese"},
        {"id": "Sohee", "name": "Sohee", "description": "Warm Korean female voice with rich emotion", "native_language": "Korean"},
    ]
    
    SUPPORTED_LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto"]
    
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
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud"
                },
                "model_type": {
                    "type": "string",
                    "enum": ["custom_voice_0.6b", "custom_voice_1.7b", "voice_design_1.7b"],
                    "description": "Model type to use",
                    "default": "custom_voice_0.6b"
                },
                "speaker": {
                    "type": "string",
                    "description": "Speaker ID for CustomVoice models (e.g., 'Vivian', 'Ryan')",
                    "default": "Vivian"
                },
                "language": {
                    "type": "string",
                    "description": "Language code (Chinese, English, Japanese, etc.) or 'Auto' for auto-detect",
                    "default": "Auto"
                },
                "instruct": {
                    "type": "string",
                    "description": "Natural language instruction for voice control (optional)",
                    "default": ""
                }
            },
            "required": ["text"]
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
        self.voice_output = self.tts_config.get('voice_output', 'local')
        self.current_audio_process = None
        self.is_playing = False
        self.current_audio_type = None
        self._model = None
        self._model_loaded = False
        self._device = qwen_config.get("device", "cuda:0")
        self._dtype = qwen_config.get("dtype", "bfloat16")
    
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
            {"id": "custom_voice_0.6b", "name": "Qwen3-TTS 0.6B CustomVoice", "description": "Faster, smaller model with predefined speakers"},
            {"id": "custom_voice_1.7b", "name": "Qwen3-TTS 1.7B CustomVoice", "description": "Higher quality model with predefined speakers"},
            {"id": "voice_design_1.7b", "name": "Qwen3-TTS 1.7B VoiceDesign", "description": "Design voices using natural language descriptions"},
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
            print(f"[Qwen TTS] Warning: Invalid speaker '{speaker}', keeping '{self.current_speaker}'")
    
    def set_language(self, language: str):
        """Set the language."""
        if language in self.SUPPORTED_LANGUAGES or language.capitalize() in self.SUPPORTED_LANGUAGES:
            self.current_language = language.capitalize() if language.capitalize() in self.SUPPORTED_LANGUAGES else language
            print(f"[Qwen TTS] Language set to: {self.current_language}")
    
    def set_instruct(self, instruct: str):
        """Set the instruction for voice control."""
        self.current_instruct = instruct
        print(f"[Qwen TTS] Instruct set to: {instruct[:50]}..." if len(instruct) > 50 else f"[Qwen TTS] Instruct set to: {instruct}")
    
    def _load_model(self):
        """Load the Qwen3-TTS model."""
        if self._model_loaded and self._model is not None:
            return True
        
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            
            model_id = self.MODEL_TYPES.get(self.model_type)
            if not model_id:
                print(f"[Qwen TTS] Error: Unknown model type '{self.model_type}'")
                return False
            
            print(f"[Qwen TTS] Loading model: {model_id}")
            
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self._dtype, torch.bfloat16)
            
            self._model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=self._device,
                dtype=dtype,
                attn_implementation="flash_attention_2",
            )
            self._model_loaded = True
            print(f"[Qwen TTS] Model loaded successfully")
            return True
            
        except ImportError as e:
            print(f"[Qwen TTS] Error: qwen-tts package not installed. Run: pip install -U qwen-tts")
            print(f"[Qwen TTS] ImportError: {e}")
            return False
        except Exception as e:
            print(f"[Qwen TTS] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _play_audio(self, audio_file: str):
        """Play audio file using system player."""
        try:
            if os.path.exists(audio_file):
                cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_file]
                proc = subprocess.Popen(cmd)
                return proc
        except Exception as e:
            print(f"[Qwen TTS] Error playing audio: {e}")
        return None
    
    def stop_audio(self, reason: str = ""):
        """Stop current audio playback."""
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
    
    async def execute(
        self,
        text: str,
        model_type: str = None,
        speaker: str = None,
        language: str = None,
        instruct: str = None,
        audio_type: str = None,
        **kwargs
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
        
        if not self._load_model():
            return {
                "success": False,
                "error": "Failed to load Qwen3-TTS model. Make sure qwen-tts is installed and model is downloaded."
            }
        
        try:
            import soundfile as sf
            
            print(f"[Qwen TTS] Generating speech for: {text[:50]}...")
            
            is_voice_design = "voice_design" in self.model_type
            lang = self.current_language if self.current_language != "Auto" else "Auto"
            
            start_time = time.time()
            
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
            print(f"[Qwen TTS] Generated in {gen_time:.2f}s")
            
            output_file = os.path.join(self.temp_dir, f"qwen_tts_{int(time.time())}.wav")
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
            return {
                "success": False,
                "error": f"Qwen TTS failed: {str(e)}"
            }
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"[Qwen TTS] Error cleaning up: {e}")
