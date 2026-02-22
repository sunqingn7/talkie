"""
LLM Model Manager for Talkie
Handles llama-server lifecycle and model switching
"""

import os
import subprocess
import signal
import time
import yaml
import glob
import psutil
from typing import Dict, List, Optional, Tuple, Tuple
from pathlib import Path


class LLMModelManager:
    """Manages llama-server processes and model switching."""
    
    def __init__(self, config_path: str = "config/models.yaml", params_path: str = "config/model_params.yaml"):
        self.config_path = config_path
        self.params_path = params_path
        self.config = self._load_config()
        self.custom_params = self._load_custom_params()
        self.current_process: Optional[subprocess.Popen] = None
        self.current_model_id: Optional[str] = None
        self.cache_dir = os.path.expanduser("~/.cache/llama.cpp")
        self._current_timeout = 120  # Default timeout in seconds
        
    def get_current_timeout(self) -> int:
        """Get the current timeout value for LLM requests."""
        return self._current_timeout
    
    def _parse_talkie_timeout(self, extra_flags: str) -> int:
        """Parse --talkie_timeout N from extra flags."""
        import shlex
        try:
            args = shlex.split(extra_flags)
            for i, arg in enumerate(args):
                if arg == "--talkie_timeout" and i + 1 < len(args):
                    return int(args[i + 1])
        except:
            pass
        return 120  # Default
    
    def set_timeout_from_extra_flags(self, model_id: Optional[str] = None):
        """Update timeout from extra flags."""
        extra_flags = self.get_extra_flags(model_id)
        self._current_timeout = self._parse_talkie_timeout(extra_flags)
        
    def _load_config(self) -> dict:
        """Load model configuration."""
        # Try multiple paths
        paths = [
            self.config_path,
            os.path.join(os.path.dirname(__file__), '..', '..', self.config_path),
            os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'models.yaml'),
        ]
        
        for path in paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
        
        # Return default config if file not found
        return {"models": {}, "global_settings": {}}
    
    def _load_custom_params(self) -> dict:
        """Load custom model parameters from model_params.yaml."""
        paths = [
            self.params_path,
            os.path.join(os.path.dirname(__file__), '..', '..', self.params_path),
            os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'model_params.yaml'),
        ]
        
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = yaml.safe_load(f)
                        # Ensure we always return a valid dict with required keys
                        if data is None:
                            data = {}
                        if "global_params" not in data:
                            data["global_params"] = {}
                        if "per_model_params" not in data:
                            data["per_model_params"] = {}
                        if "extra_params" not in data:
                            data["extra_params"] = ""
                        return data
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")
        
        return {"global_params": {}, "per_model_params": {}, "extra_params": ""}
    
    def save_custom_params(self) -> bool:
        """Save custom parameters to model_params.yaml."""
        try:
            path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'model_params.yaml')
            with open(path, 'w') as f:
                yaml.dump(self.custom_params, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            return True
        except Exception as e:
            print(f"Failed to save custom params: {e}")
            return False
    
    def get_custom_params(self) -> dict:
        """Get all custom parameters."""
        return self.custom_params
    
    def set_global_params(self, params: dict) -> bool:
        """Set global parameters that apply to all models."""
        # Ensure params is a valid dict, not None
        if params is None:
            params = {}
        self.custom_params["global_params"] = params
        return self.save_custom_params()
    
    def set_model_params(self, model_id: str, params: dict) -> bool:
        """Set custom parameters for a specific model."""
        if "per_model_params" not in self.custom_params:
            self.custom_params["per_model_params"] = {}
        if params and len(params) > 0:
            self.custom_params["per_model_params"][model_id] = params
        elif model_id in self.custom_params["per_model_params"]:
            del self.custom_params["per_model_params"][model_id]
        return self.save_custom_params()
    
    def set_extra_params(self, extra: str) -> bool:
        """Set extra command-line parameters."""
        self.custom_params["extra_params"] = extra
        return self.save_custom_params()
    
    def get_merged_params(self, model_id: str, base_params: dict) -> dict:
        """Merge base params with global and per-model custom params."""
        merged = dict(base_params)
        
        # Apply global params (handle None case)
        global_params = self.custom_params.get("global_params") or {}
        for key, value in global_params.items():
            if key != "extra_flags":  # extra_flags handled separately
                merged[key] = value
        
        # Apply per-model params (takes precedence)
        per_model_params = self.custom_params.get("per_model_params") or {}
        per_model = per_model_params.get(model_id) or {}
        for key, value in per_model.items():
            if key != "extra_flags":
                merged[key] = value
        
        return merged
    
    def get_extra_flags(self, model_id: Optional[str] = None) -> str:
        """Get combined extra flags from global and per-model params."""
        flags = []
        
        global_params = self.custom_params.get("global_params") or {}
        global_extra = global_params.get("extra_flags", "")
        if global_extra:
            flags.append(global_extra)
        
        if model_id:
            per_model_params = self.custom_params.get("per_model_params") or {}
            model_params = per_model_params.get(model_id) or {}
            model_extra = model_params.get("extra_flags", "")
            if model_extra:
                flags.append(model_extra)
        
        extra_params = self.custom_params.get("extra_params", "")
        if extra_params:
            flags.append(extra_params)
        
        return " ".join(flags)
    
    def scan_available_models(self) -> List[Dict]:
        """Scan cache directory for available GGUF models."""
        available_models = []
        
        # Get configured models
        configured_models = self.config.get("models", {})
        
        # Track sharded models by their base name
        sharded_models = {}  # base_name -> list of shard files
        
        # Scan cache directory
        cache_path = Path(self.cache_dir)
        if cache_path.exists():
            # First pass: collect all GGUF files
            for gguf_file in cache_path.glob("*.gguf"):
                # Skip incomplete downloads and etag files
                if ".downloadInProgress" in gguf_file.name or gguf_file.name.endswith(".etag"):
                    continue
                
                # Check if this is a shard (contains -NNNNN-of-NNNNN pattern)
                import re
                shard_match = re.search(r'-(\d+)-of-(\d+)\.gguf$', gguf_file.name)
                if shard_match:
                    # This is a sharded model
                    # Remove the shard suffix: "-00001-of-00004.gguf" -> ""
                    base_name = gguf_file.name[:-(len(shard_match.group(0)))]
                    if base_name not in sharded_models:
                        sharded_models[base_name] = []
                    sharded_models[base_name].append(gguf_file)
                else:
                    # Single file model - process immediately
                    self._process_gguf_file(gguf_file, configured_models, available_models)
            
            # Process sharded models
            for base_name, shard_files in sharded_models.items():
                # Sort shards by name to ensure correct order
                shard_files.sort(key=lambda x: x.name)
                
                # Calculate total size
                total_size = sum(f.stat().st_size for f in shard_files)
                total_size_gb = total_size / (1024**3)
                
                # Find matching config
                model_id = None
                model_config = None
                
                for mid, mconfig in configured_models.items():
                    if base_name in (mconfig.get("file", "")):
                        model_id = mid
                        model_config = mconfig
                        break
                
                # If no config found, create basic info
                if not model_config:
                    # Extract nice name from base_name
                    # Handle patterns like "unsloth_MiniMax-M2.5-GGUF_MXFP4_MOE_MiniMax-M2.5-MXFP4_MOE"
                    # Try to extract a cleaner name
                    nice_name = base_name
                    # Remove common prefixes
                    for prefix in ['unsloth_', 'gguf_', 'model_']:
                        if nice_name.lower().startswith(prefix):
                            nice_name = nice_name[len(prefix):]
                    # Replace separators with spaces
                    nice_name = nice_name.replace("_", " ").replace("-", " ").strip()
                    # Remove duplicate words (case insensitive)
                    words = nice_name.split()
                    unique_words = []
                    seen_lower = set()
                    for word in words:
                        if word.lower() not in seen_lower:
                            unique_words.append(word)
                            seen_lower.add(word.lower())
                    nice_name = " ".join(unique_words)
                    # Title case
                    nice_name = nice_name.title()
                    
                    model_id = base_name.replace("_", "-").lower()
                    model_config = {
                        "name": nice_name,
                        "description": f"Sharded model: {len(shard_files)} files, {total_size_gb:.1f}GB total",
                        "file": base_name,
                        "shard_files": [f.name for f in shard_files],
                        "size": f"{total_size_gb:.1f}GB",
                        "sharded": True,
                        "quantization": "Unknown",
                        "recommended_for": ["general"],
                        "llama_server_params": self._get_default_params()
                    }
                
                available_models.append({
                    "id": model_id,
                    **model_config,
                    "path": str(shard_files[0].parent / base_name),  # Path to first shard
                    "exists": True
                })
        
        # Also add configured models that don't exist yet
        for model_id, model_config in configured_models.items():
            if not any(m["id"] == model_id for m in available_models):
                available_models.append({
                    "id": model_id,
                    **model_config,
                    "path": None,
                    "exists": False
                })
        
        return available_models
    
    def _process_gguf_file(self, gguf_file, configured_models, available_models):
        """Process a single GGUF file."""
        # Find matching config
        model_id = None
        model_config = None
        
        for mid, mconfig in configured_models.items():
            if mconfig.get("file") == gguf_file.name:
                model_id = mid
                model_config = mconfig
                break
        
        # If config found, use it
        if model_config:
            available_models.append({
                "id": model_id,
                **model_config,
                "path": str(gguf_file),
                "exists": True
            })
        else:
            # If no config found, create basic info
            model_id = gguf_file.stem.replace("_", "-").lower()
            size_gb = gguf_file.stat().st_size / (1024**3)
            model_config = {
                "name": gguf_file.stem.replace("_", " ").replace("-", " ").title(),
                "description": f"Model file: {gguf_file.name}",
                "file": gguf_file.name,
                "size": f"{size_gb:.1f}GB",
                "quantization": "Unknown",
                "recommended_for": ["general"],
                "llama_server_params": self._get_default_params()
            }
            
            available_models.append({
                "id": model_id,
                **model_config,
                "path": str(gguf_file),
                "exists": True
            })
    
    def _get_default_params(self) -> dict:
        """Get default llama-server parameters."""
        return {
            "ctx_size": 8192,
            "threads": 8,
            "threads_batch": 8,
            "n_gpu_layers": -1,
            "batch_size": 2048,
            "parallel": 1,
            "host": "localhost",
            "port": 8080,
            "mlock": False,
            "no_mmap": False,
            "cont_batching": True,
            "flash_attn": True,
            "embeddings": True
        }
    
    def get_running_model(self) -> Optional[Dict]:
        """Get information about currently running model."""
        # Check if llama-server is running
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check for llama-server process name (could be full path or just 'llama-server')
                proc_name = proc.info['name'] or ''
                if proc_name == 'llama-server' or 'llama-server' in proc_name:
                    cmdline = proc.info['cmdline'] or []
                    
                    # Extract model info from command line
                    model_path = None
                    for i, arg in enumerate(cmdline):
                        if arg in ['-m', '--model'] and i + 1 < len(cmdline):
                            model_path = cmdline[i + 1]
                            break
                        elif arg.startswith('-m'):
                            model_path = arg[2:]
                            break
                    
                    # Also check for -hf or --hf flag (HuggingFace download)
                    if not model_path:
                        for i, arg in enumerate(cmdline):
                            if arg in ['-hf', '--hf'] and i + 1 < len(cmdline):
                                model_path = cmdline[i + 1]
                                break
                            elif arg.startswith('-hf') and len(arg) > 3:
                                model_path = arg[3:]  # -hf model_name
                                break
                            elif arg.startswith('--hf='):
                                model_path = arg[5:]  # --hf=model_name
                                break
                    
                    # Extract a friendly model name
                    model_name = model_path or ""
                    if '/' in model_name:
                        # Handle HuggingFace model IDs like "unsloth/MiniMax-M2.5-GGUF:MXFP4_MOE"
                        # Get the repo/model name part (after last /)
                        hf_model = model_name.split('/')[-1]
                        # Remove quantization suffix after : if present
                        if ':' in hf_model:
                            hf_model = hf_model.split(':')[0]
                        # Remove -GGUF suffix if present
                        if hf_model.endswith('-GGUF'):
                            hf_model = hf_model[:-5]
                        model_name = hf_model.replace("_", " ").replace("-", " ").strip()
                        # Title case but preserve version numbers
                        words = model_name.split()
                        model_name = " ".join(w.title() if not any(c.isdigit() for c in w) else w for w in words)
                    
                    return {
                        "pid": proc.info['pid'],
                        "model_path": model_path,
                        "model_name": model_name,
                        "cmdline": cmdline,
                        "running": True
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return None
    
    def stop_server(self) -> bool:
        """Stop the running llama-server."""
        running = self.get_running_model()
        
        if not running:
            print("⚠️  No llama-server process found")
            return True
        
        pid = running["pid"]
        print(f"🛑 Stopping llama-server (PID: {pid})...")
        
        try:
            # Try graceful termination first
            os.kill(pid, signal.SIGTERM)
            
            # Wait for process to terminate
            for _ in range(10):  # Wait up to 5 seconds
                if not psutil.pid_exists(pid):
                    print("✅ llama-server stopped gracefully")
                    return True
                time.sleep(0.5)
            
            # Force kill if still running
            if psutil.pid_exists(pid):
                print("⚠️  Force stopping llama-server...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
            
            if not psutil.pid_exists(pid):
                print("✅ llama-server stopped")
                return True
            else:
                print("❌ Failed to stop llama-server")
                return False
                
        except Exception as e:
            print(f"❌ Error stopping server: {e}")
            return False
    
    def start_server(self, model_id: str, wait_for_ready: bool = True) -> Tuple[bool, str]:
        """Start llama-server with specified model."""
        try:
            # Get model configuration
            models = self.scan_available_models()
            model = next((m for m in models if m["id"] == model_id), None)
            
            if not model:
                return False, f"Model '{model_id}' not found"
            
            if not model.get("exists"):
                return False, f"Model file not found: {model.get('file')}"
            
            # Stop any running server
            if not self.stop_server():
                return False, "Failed to stop existing server"
            
            # Build command
            binary_path = self.config.get("global_settings", {}).get(
                "binary_path", 
                "/home/qing/Project/llama.cpp/build/bin/llama-server"
            )
            
            if not os.path.exists(binary_path):
                return False, f"llama-server binary not found: {binary_path}"
            
            base_params = model.get("llama_server_params") or self._get_default_params()
            params = self.get_merged_params(model_id, base_params)
            
            cmd = [binary_path]
            
            # Add model
            cmd.extend(["-m", model["path"]])
            
            # Add parameters
            if params.get("ctx_size"):
                cmd.extend(["-c", str(params["ctx_size"])])
            
            if params.get("threads"):
                cmd.extend(["-t", str(params["threads"])])
            
            if params.get("threads_batch"):
                cmd.extend(["-tb", str(params["threads_batch"])])
            
            if params.get("n_gpu_layers") is not None:
                cmd.extend(["-ngl", str(params["n_gpu_layers"])])
            
            if params.get("batch_size"):
                cmd.extend(["-b", str(params["batch_size"])])
            
            if params.get("parallel"):
                cmd.extend(["--parallel", str(params["parallel"])])
            
            if params.get("host"):
                cmd.extend(["--host", params["host"]])
            
            if params.get("port"):
                cmd.extend(["--port", str(params["port"])])
            
            if params.get("mlock"):
                cmd.append("--mlock")
            
            if params.get("no_mmap"):
                cmd.append("--no-mmap")
            
            if params.get("cont_batching"):
                cmd.append("--cont-batching")
            
            if params.get("flash_attn"):
                cmd.extend(["--flash-attn", "on"])
            
            if params.get("embeddings"):
                cmd.append("--embeddings")
            
            if params.get("numa"):
                cmd.extend(["--numa", params["numa"]])
            
            if params.get("main_gpu") is not None:
                cmd.extend(["--main-gpu", str(params["main_gpu"])])
            
            if params.get("tensor_split"):
                cmd.extend(["-ts", params["tensor_split"]])
            
            # Enable Jinja templating for chat formats
            cmd.append("--jinja")
            
            # Add extra custom flags (with conflict detection and replacement)
            extra_flags = self.get_extra_flags(model_id)
            if extra_flags:
                import shlex
                extra_args = shlex.split(extra_flags)
                
                # Parse extra args into dict: {flag: value_or_None}
                extra_dict = {}
                i = 0
                while i < len(extra_args):
                    arg = extra_args[i]
                    # Handle --talkie_timeout specially (talkie internal, not passed to llama-server)
                    if arg == "--talkie_timeout":
                        if i + 1 < len(extra_args):
                            extra_dict[arg] = extra_args[i + 1]
                            i += 2
                        else:
                            extra_dict[arg] = None
                            i += 1
                    # Determine if this arg expects a value
                    elif arg in ("-c", "-t", "-tb", "-ngl", "-b", "--parallel", "--host", "--port",
                               "--flash-attn", "--numa", "--main-gpu", "-ts"):
                        if i + 1 < len(extra_args):
                            extra_dict[arg] = extra_args[i + 1]
                            i += 2
                        else:
                            extra_dict[arg] = None
                            i += 1
                    else:
                        # Boolean flags or unknown flags
                        extra_dict[arg] = None
                        i += 1
                
                # Define which params can be replaced (including --talkie_timeout which is internal)
                replaceable_params = {
                    "-c", "-t", "-tb", "-ngl", "-b", "--parallel", "--host", "--port",
                    "--flash-attn", "--numa", "--main-gpu", "-ts", "--mlock", "--no-mmap",
                    "--cont-batching", "--embeddings", "--talkie_timeout"
                }
                
                # Build new cmd: skip params that will be replaced
                new_cmd = [cmd[0]]  # Keep binary path
                i = 1
                while i < len(cmd):
                    if cmd[i] in replaceable_params and cmd[i] in extra_dict:
                        # Skip this param and its value if it has one
                        if cmd[i] not in ("--mlock", "--no-mmap", "--cont-batching", "--embeddings", "--jinja"):
                            i += 2  # Skip flag and value
                        else:
                            i += 1  # Skip boolean flag
                    else:
                        new_cmd.append(cmd[i])
                        i += 1
                
                # Add extra args
                for arg, value in extra_dict.items():
                    new_cmd.append(arg)
                    if value is not None:
                        new_cmd.append(value)
                
                cmd = new_cmd
            
            # Update timeout from extra flags
            self.set_timeout_from_extra_flags(model_id)
            
            print(f"🚀 Starting llama-server with model: {model['name']}")
            print(f"   Command: {' '.join(cmd)}")
            
            # Start server process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Create new process group
            )
            
            self.current_process = process
            self.current_model_id = model_id
            
            if wait_for_ready:
                print("   Waiting for server to be ready...")
                if self._wait_for_server_ready(params.get("host", "localhost"), params.get("port", 8080)):
                    print(f"✅ Server ready! Model: {model['name']}")
                    return True, f"Server started with model: {model['name']}"
                else:
                    return False, "Server failed to start within timeout"
            else:
                return True, "Server started (not waiting for ready)"
                
        except Exception as e:
            import traceback
            print(f"❌ Error starting server: {e}")
            traceback.print_exc()
            return False, f"Failed to start server: {str(e)}"
    
    def _wait_for_server_ready(self, host: str, port: int, timeout: int = 60) -> bool:
        """Wait for llama-server to be ready."""
        import urllib.request
        import json
        
        start_time = time.time()
        url = f"http://{host}:{port}/health"
        
        while time.time() - start_time < timeout:
            try:
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        return True
            except:
                pass
            
            # Also check if process is still running
            if self.current_process and self.current_process.poll() is not None:
                # Process exited
                stdout, stderr = self.current_process.communicate()
                print(f"❌ Server process exited with code: {self.current_process.returncode}")
                if stderr:
                    print(f"   Error: {stderr[:500]}")
                return False
            
            time.sleep(0.5)
        
        return False
    
    def switch_model(self, model_id: str) -> Dict:
        """Switch to a different model."""
        print(f"🔄 Switching to model: {model_id}")
        
        success, message = self.start_server(model_id)
        
        return {
            "success": success,
            "message": message,
            "model_id": model_id if success else None,
            "timestamp": time.time()
        }
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get detailed information about a model."""
        models = self.scan_available_models()
        return next((m for m in models if m["id"] == model_id), None)
    
    def restart_server(self) -> Dict:
        """Restart the current server."""
        if not self.current_model_id:
            running = self.get_running_model()
            if running and running.get("model_path"):
                # Try to extract model ID from path
                for model in self.scan_available_models():
                    if model.get("path") == running["model_path"]:
                        self.current_model_id = model["id"]
                        break
        
        if self.current_model_id:
            return self.switch_model(self.current_model_id)
        else:
            return {
                "success": False,
                "message": "No model currently running",
                "timestamp": time.time()
            }


# Singleton instance
_model_manager = None

def get_model_manager(config_path: str = "config/models.yaml") -> LLMModelManager:
    """Get singleton instance of model manager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = LLMModelManager(config_path)
    return _model_manager
