import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
import json

import modal

# --- Configuration ---
DEFAULT_MODEL = "DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B-GGUF"

# --- Modal App Setup ---
app = modal.App("llamacpp-openai-server")

# --- Create a default volume for the app ---
default_volume = modal.Volume.from_name("llamacpp-models-storage", create_if_missing=True)

# --- Helper functions ---
def get_model_path_from_name(model_name: str):
    """Get a safe path for the model inside the volume"""
    # Create a safe directory name from model name
    safe_name = model_name.replace("/", "--").replace("_", "-").replace(".", "-").lower()
    model_base_path = Path("/models")
    model_path = model_base_path / safe_name
    return model_path

def get_model_config(model_path: Path):
    """Read model config to get actual limits"""
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract key parameters
            max_position_embeddings = config.get("max_position_embeddings", None)
            model_max_length = config.get("model_max_length", None) or config.get("max_sequence_length", None)
            vocab_size = config.get("vocab_size", None)
            hidden_size = config.get("hidden_size", None)
            model_type = config.get("model_type", "unknown")
            architectures = config.get("architectures", [])
            
            print(f"📖 Model config: type={model_type}, arch={architectures}, max_pos_emb={max_position_embeddings}, max_len={model_max_length}")
            
            return {
                "max_position_embeddings": max_position_embeddings,
                "model_max_length": model_max_length,
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "model_type": model_type,
                "architectures": architectures
            }
        except Exception as e:
            print(f"⚠️ Could not read model config: {e}")
            return None
    return None

def is_gguf_model(model_name: str) -> bool:
    """Check if the model is a GGUF model"""
    return "gguf" in model_name.lower()

def get_gguf_file_path(model_path: Path) -> Path:
    """Get the best GGUF file (prefer Q8_0, then highest quality)"""
    gguf_files = list(model_path.glob("*.gguf"))
    
    if len(gguf_files) == 0:
        raise ValueError("No GGUF files found")
    elif len(gguf_files) == 1:
        print(f"🎯 Using single GGUF file: {gguf_files[0].name}")
        return gguf_files[0]
    else:
        print(f"📊 Found {len(gguf_files)} GGUF files, selecting best quality...")
        
        # Priority order: Q8_0 (best) -> Q6_K -> Q5_K_M -> Q4_K_M -> others
        priority_order = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"]
        
        for priority in priority_order:
            for gguf_file in gguf_files:
                if priority in gguf_file.name:
                    size_gb = gguf_file.stat().st_size / 1e9
                    print(f"🎯 Selected GGUF file: {gguf_file.name} ({size_gb:.1f} GB)")
                    print(f"🗜️ Quantization level: {priority}")
                    return gguf_file
        
        # Fallback to largest file (usually highest quality)
        largest_file = max(gguf_files, key=lambda f: f.stat().st_size)
        size_gb = largest_file.stat().st_size / 1e9
        print(f"⚠️ Using fallback (largest) GGUF file: {largest_file.name} ({size_gb:.1f} GB)")
        return largest_file

def download_model_to_path(model_name: str, model_path: Path):
    """Download model to specific path with GGUF optimization"""
    import os
    import shutil
    from huggingface_hub import snapshot_download
    
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Check if model already exists and is complete
    if model_path.exists():
        print(f"📁 Model directory exists at {model_path}, checking integrity...")
        
        # For GGUF models, check for GGUF file
        if is_gguf_model(model_name):
            gguf_files = list(model_path.glob("*.gguf"))
            if len(gguf_files) == 0:
                print(f"❌ No GGUF files found, re-downloading...")
                shutil.rmtree(model_path)
            else:
                print("✅ GGUF model integrity check passed - using cached model")
                return
        else:
            print("❌ Non-GGUF models not supported in llama-cpp mode")
            return

    if not model_path.exists():
        print(f"📥 Downloading {model_name} to {model_path}...")
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # For GGUF models, use selective download to get only the best quality file
            if is_gguf_model(model_name):
                print(f"🎯 GGUF model detected - downloading selectively...")
                # Download only Q8_0 (best quality) and essential files
                snapshot_download(
                    repo_id=model_name,
                    local_dir=model_path,
                    allow_patterns=[
                        "*Q8_0.gguf",      # Best quality GGUF
                        "*Q6_K.gguf",      # Fallback 1
                        "*Q5_K_M.gguf",    # Fallback 2
                        "*.json",          # Config files
                        "*.txt",           # README, etc.
                        "*.md",            # Documentation
                        "tokenizer*",      # Tokenizer files
                        "*.model",         # SentencePiece model files
                        "*.vocab",         # Vocabulary files
                        "special_tokens_map.json",  # Special tokens
                        "tokenizer_config.json",    # Tokenizer config
                        "vocab.json",               # Vocab JSON
                        "merges.txt",              # BPE merges
                    ],
                    token=os.environ.get("HF_TOKEN"),
                    resume_download=True,
                    local_files_only=False,
                )
                print(f"🎯 Selective GGUF download completed - only high-quality quantizations downloaded")
                
                # Check if we have tokenizer files, if not, try to get them from base model
                tokenizer_files = list(model_path.glob("tokenizer*")) + list(model_path.glob("*.model"))
                if len(tokenizer_files) == 0:
                    print(f"⚠️ No tokenizer files found in GGUF repo, trying to get from base model...")
                    # Try to extract base model name and download tokenizer from there
                    base_model_candidates = [
                        "meta-llama/Llama-3.2-3B-Instruct",  # Common base for Llama 3.2
                        "meta-llama/Meta-Llama-3-8B-Instruct",  # Fallback
                    ]
                    
                    for base_model in base_model_candidates:
                        try:
                            print(f"🔄 Trying to get tokenizer from {base_model}...")
                            snapshot_download(
                                repo_id=base_model,
                                local_dir=model_path,
                                allow_patterns=[
                                    "tokenizer*",
                                    "*.model",
                                    "*.vocab",
                                    "special_tokens_map.json",
                                    "tokenizer_config.json",
                                    "vocab.json",
                                    "merges.txt",
                                ],
                                token=os.environ.get("HF_TOKEN"),
                                resume_download=True,
                                local_files_only=False,
                            )
                            print(f"✅ Tokenizer files downloaded from {base_model}")
                            break
                        except Exception as e:
                            print(f"⚠️ Failed to get tokenizer from {base_model}: {e}")
                            continue
            else:
                print("❌ Only GGUF models are supported in llama-cpp mode")
                return
            
            print("✅ Download complete!")
            
            # Check what we downloaded
            gguf_files = list(model_path.glob("*.gguf"))
            print(f"📊 Found {len(gguf_files)} GGUF files")
            tokenizer_files = list(model_path.glob("tokenizer*")) + list(model_path.glob("*.model"))
            print(f"🔤 Found {len(tokenizer_files)} tokenizer files")
            
            print("💾 Model saved to persistent volume!")
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if model_path.exists():
                shutil.rmtree(model_path)
            raise
    else:
        print(f"✅ Model already exists at {model_path}")

# --- GPU selection based on model (adapted for llama-cpp) ---
def get_gpu_config_llamacpp(model_name: str):
    """Get appropriate GPU configuration for llama-cpp"""
    model_name_lower = model_name.lower()
    
    # GGUF MoE models (special handling - no special MoE flags needed)
    if is_gguf_model(model_name):
        if "moe" in model_name_lower or "8x3b" in model_name_lower:
            if "18b" in model_name_lower or "18.4b" in model_name_lower:
                return "H100", 0.95  # 18.4B MoE GGUF needs H100
            elif "14b" in model_name_lower:
                return "H100", 0.90  # 14B MoE GGUF
            elif "7b" in model_name_lower or "8b" in model_name_lower:
                return "L40S", 0.85  # 7-8B MoE GGUF
        # Regular GGUF models
        elif "34b" in model_name_lower:
            return "H200", 0.85  # 34B GGUF
        elif "13b" in model_name_lower or "14b" in model_name_lower:
            return "A100-80GB", 0.90  # 13-14B GGUF
        elif "7b" in model_name_lower or "8b" in model_name_lower:
            return "L40S", 0.85  # 7-8B GGUF
        elif "3b" in model_name_lower:
            return "A10G", 0.85  # 3B GGUF
        else:
            return "L4", 0.85  # Small GGUF models
    
    # Default for non-GGUF (though not supported)
    return "L4", 0.85

def get_llamacpp_config(model_name: str, gpu_type: str, gguf_file_path: Path):
    """Get llama-cpp server configuration"""
    config = {
        "n_ctx": 2048,
        "n_batch": 512,
        "n_gpu_layers": -1,  # All layers to GPU by default
        "use_mmap": True,
        "use_mlock": False,  # False for Modal environment
        "chat_format": "chatml",  # RP-friendly default
        "verbose": True,
        "n_threads": 8
    }
    
    model_lower = model_name.lower()
    
    # GPU-specific optimizations
    if gpu_type in ["H100", "H200", "B200"]:
        config.update({
            "n_ctx": 4096,
            "n_batch": 1024,
            "n_threads": 16
        })
    elif gpu_type in ["A100-80GB", "L40S"]:
        config.update({
            "n_ctx": 2048,
            "n_batch": 512,
            "n_threads": 12
        })
    elif gpu_type in ["L4", "A10G"]:
        config.update({
            "n_ctx": 1024,
            "n_batch": 256,
            "n_threads": 8
        })
    
    # Model-specific adjustments
    if "34b" in model_lower:
        config["n_ctx"] = min(config["n_ctx"], 1024)  # Conservative for 34B
        config["n_batch"] = min(config["n_batch"], 256)
    elif "moe" in model_lower:
        config["n_gpu_layers"] = 20  # Partial GPU offload for MoE
        config["n_ctx"] = min(config["n_ctx"], 2048)  # Conservative for MoE
    
    # Chat format selection
    if "llama" in model_lower:
        config["chat_format"] = "llama-2"
    elif "mistral" in model_lower:
        config["chat_format"] = "mistral-instruct"
    elif "yi" in model_lower or "qwen" in model_lower:
        config["chat_format"] = "chatml"
    
    return config

# --- Container image with llama-cpp ---
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "llama-cpp-python[server]==0.3.15",  # Latest version with excellent MoE support
        "fastapi",
        "uvicorn", 
        "pydantic",
        "huggingface_hub[hf_transfer]",
        "requests",
        "psutil",
        "prometheus_client"
    )
)

# --- Core chat function with llama-cpp ---
def run_llamacpp_logic(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    """Core llama-cpp logic that runs the server and handles chat"""
    if not is_gguf_model(model_name):
        raise ValueError("Only GGUF models are supported in llama-cpp mode")
    
    gpu_config = get_gpu_config_llamacpp(model_name)
    gpu_type, memory_util = gpu_config
    moe_flag = ""  # No special MoE flags needed for llama-cpp
    
    model_path = get_model_path_from_name(model_name)
    
    port = 8000
    
    print(f"🤖 Using model: {model_name}")
    print(f"🔧 GPU: {gpu_type}, Memory utilization: {memory_util}")
    print(f"📁 Model path: {model_path}")
    print(f"📦 Using persistent volume for model storage")
    print(f"🦙 Engine: llama-cpp-python v0.3.15")
    
    # Download model if it doesn't exist
    if not model_path.exists():
        print(f"📥 Model not found in volume, downloading...")
        download_model_to_path(model_name, model_path)
    else:
        print(f"✅ Model found in volume: {model_path}")
    
    if not model_path.exists():
        raise RuntimeError(f"Model path {model_path} does not exist after download")
    
    # Get GGUF file
    gguf_file_path = get_gguf_file_path(model_path)
    size_gb = gguf_file_path.stat().st_size / 1e9
    print(f"📊 Using GGUF file: {gguf_file_path.name} ({size_gb:.1f} GB)")
    
    # Get configuration
    config = get_llamacpp_config(model_name, gpu_type, gguf_file_path)
    
    # Start llama-cpp server
    llamacpp_process = run_llamacpp_server(gguf_file_path, config, moe_flag, port)
    
    # Wait for startup (MoE models need more time) with real-time logging
    wait_for_server_ready(port, max_wait=180, process=llamacpp_process)
    
    # Server ready - continue with API/chat logic
    with modal.forward(port) as tunnel:
        print(f"🌐 Server URL: {tunnel.url}")
        
        # Test API
        import requests
        try:
            test_response = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
            print(f"✅ API test successful: {test_response.status_code}")
            
            models_data = test_response.json()
            available_models = [m.get('id', 'Unknown') for m in models_data.get('data', [])]
            print(f"📋 Available models: {', '.join(available_models)}")
        except Exception as e:
            print(f"⚠️  API test failed: {e}")
        
        # Handle different modes
        if api_only:
            print("\n🌐 API Server ready! Running in API-only mode.")
            print(f"📖 Server URL: {tunnel.url}")
            print("🔌 Available endpoints:")
            print(f"  - Health: {tunnel.url}/health")
            print(f"  - Models: {tunnel.url}/v1/models")
            print(f"  - Chat: {tunnel.url}/v1/chat/completions")
            print(f"  - Completions: {tunnel.url}/v1/completions")
            print(f"💾 Model size: {size_gb:.1f} GB on {gpu_type}")
            print(f"💬 Chat format: {config['chat_format']}")
            print(f"🗜️ Quantization: GGUF built-in")
            if moe_flag:
                print(f"🔀 MoE optimization: {moe_flag}")
            print("\n⏰ Server running indefinitely. Modal will auto-scale down after inactivity.")
            print("💡 Press Ctrl+C to stop the server manually.")
            
            try:
                while True:
                    time.sleep(300)
                    print("💓 Server heartbeat - still running...")
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                return
            finally:
                llamacpp_process.terminate()
        
        elif custom_questions is None or len(custom_questions) == 0:
            print("\n🧪 API demo mode - server will stay alive for 5 minutes for testing")
            print(f"📖 Server URL: {tunnel.url}")
            print("🔌 Available endpoints:")
            print(f"  - Health: {tunnel.url}/health")
            print(f"  - Models: {tunnel.url}/v1/models")
            print(f"  - Completions: {tunnel.url}/v1/completions")
            print(f"  - Chat: {tunnel.url}/v1/chat/completions")
            print(f"💬 Chat format: {config['chat_format']}")
            print(f"🗜️ Quantization: GGUF built-in")
            
            print("\n⏰ Keeping server alive for 5 minutes for testing...")
            try:
                time.sleep(300)
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
            finally:
                llamacpp_process.terminate()
            return
        
        else:
            # Chat demo mode 
            questions = custom_questions or [
                "Hello! Please introduce yourself briefly.",
                "What can you help me with today?", 
                "Thank you for the demo!"
            ]
            
            conversation = []
            print(f"\n{'='*60}")
            print(f"🤖 Chat Session with {model_name}")
            print(f"📊 Model: {size_gb:.1f} GB on {gpu_type}")
            print(f"💬 Chat format: {config['chat_format']}")
            print(f"🗜️ Quantization: GGUF built-in")
            if moe_flag:
                print(f"🔀 MoE optimization: {moe_flag}")
            print(f"{'='*60}\n")
            
            for question in questions:
                print(f"👤 You: {question}")
                conversation.append({"role": "user", "content": question})
                
                try:
                    response = requests.post(
                        f"http://localhost:{port}/v1/chat/completions",
                        headers={"Content-Type": "application/json"},
                        json={
                            "model": gguf_file_path.name,
                            "messages": conversation,
                            "max_tokens": min(200, config["n_ctx"] // 6),
                            "temperature": 0.8,
                            "top_p": 0.9,
                        },
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        ai_response = response.json()["choices"][0]["message"]["content"]
                        conversation.append({"role": "assistant", "content": ai_response})
                        print(f"🤖 AI: {ai_response}\n")
                    else:
                        print(f"❌ Error: {response.status_code} - {response.text}\n")
                        
                except Exception as e:
                    print(f"❌ Error: {e}\n")
                
                time.sleep(1)
            
            print("✅ Chat session completed!")
            print(f"🌐 Server is still running at: {tunnel.url}")
            print(f"💬 Ready for RP proxy connections!")
            print(f"🗜️ Using GGUF built-in quantization")
            
            print("\n⏰ Keeping server alive for 5 minutes for additional testing...")
            try:
                time.sleep(300)
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
            finally:
                llamacpp_process.terminate()

def run_llamacpp_server(gguf_file_path: Path, config: dict, moe_flag: str = "", port: int = 8000):
    """Start llama-cpp OpenAI-compatible server with real-time logging"""
    command = [
        "python", "-m", "llama_cpp.server",
        "--model", str(gguf_file_path),
        "--host", "0.0.0.0",
        "--port", str(port),
        "--n_ctx", str(config["n_ctx"]),
        "--n_batch", str(config["n_batch"]),
        "--n_gpu_layers", str(config["n_gpu_layers"]),
        "--chat_format", config["chat_format"]
    ]
    
    # Optional flags
    if config.get("use_mmap"):
        command.extend(["--use_mmap", "true"])
    if config.get("use_mlock"):
        command.extend(["--use_mlock", "true"])
    
    # MoE flag if provided
    if moe_flag:
        command.extend(moe_flag.split())
    
    # Environment setup
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "GGML_CUDA_ENABLE": "1",
        "GGML_CUDA_F16": "1"  # FP16 for better performance
    })
    
    print(f"🚀 Starting llama-cpp server...")
    print(f"📁 Model: {gguf_file_path.name}")
    print(f"📊 Model size: {gguf_file_path.stat().st_size / 1e9:.1f} GB")
    print(f"⚙️ Config: ctx={config['n_ctx']}, batch={config['n_batch']}, gpu_layers={config['n_gpu_layers']}")
    print(f"💬 Chat format: {config['chat_format']}")
    if moe_flag:
        print(f"🔀 MoE optimization: {moe_flag}")
    print(f"🔧 Command: {' '.join(command)}")
    print(f"🌍 Environment vars: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
    print(f"🌍 GGML_CUDA_ENABLE={env.get('GGML_CUDA_ENABLE')}, GGML_CUDA_F16={env.get('GGML_CUDA_F16')}")
    
    # Start server process
    process = subprocess.Popen(
        command, 
        env=env, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1
    )
    
    print(f"📦 Server process started (PID: {process.pid})")
    print(f"📄 Server logs will stream below:")
    print("=" * 60)
    
    return process

def wait_for_server_ready(port: int = 8000, max_wait: int = 120, process=None):
    """Wait for llama-cpp server to be ready with real-time log streaming"""
    print(f"⏳ Waiting for llama-cpp server to start (max {max_wait}s)...")
    
    import threading
    import queue
    
    # Queue for log messages
    log_queue = queue.Queue()
    log_thread = None
    
    # Start log reading thread if process is provided
    if process:
        def read_logs():
            while True:
                try:
                    line = process.stdout.readline()
                    if not line:
                        break
                    log_queue.put(line.strip())
                except:
                    break
        
        log_thread = threading.Thread(target=read_logs, daemon=True)
        log_thread.start()
        print("📄 Starting real-time log streaming...")
    
    for i in range(max_wait):
        # Check for new log messages
        while not log_queue.empty():
            try:
                log_line = log_queue.get_nowait()
                if log_line:
                    print(f"🦙 SERVER: {log_line}")
            except queue.Empty:
                break
        
        # Check if server is ready
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print("✅ llama-cpp server is ready!")
            
            # Read any remaining logs
            if process:
                for _ in range(10):  # Read up to 10 more log lines
                    try:
                        log_line = log_queue.get_nowait()
                        if log_line:
                            print(f"🦙 SERVER: {log_line}")
                    except queue.Empty:
                        break
            
            return True
        except:
            if i % 15 == 0 and i > 0:
                print(f"⏳ Still waiting... ({i}s/{max_wait}s)")
                # Check process status
                if process and process.poll() is not None:
                    print(f"❌ Server process ended with code: {process.returncode}")
                    # Read final logs
                    while not log_queue.empty():
                        try:
                            log_line = log_queue.get_nowait()
                            if log_line:
                                print(f"🦙 SERVER: {log_line}")
                        except queue.Empty:
                            break
                    raise RuntimeError(f"Server process died with exit code {process.returncode}")
            time.sleep(1)
    
    # Timeout reached - show final status
    print(f"⏰ Timeout reached after {max_wait}s")
    if process:
        if process.poll() is None:
            print(f"📦 Server process is still running (PID: {process.pid})")
        else:
            print(f"❌ Server process ended with code: {process.returncode}")
        
        # Read final logs
        print("📄 Final server logs:")
        while not log_queue.empty():
            try:
                log_line = log_queue.get_nowait()
                if log_line:
                    print(f"🦙 SERVER: {log_line}")
            except queue.Empty:
                break
    
    raise RuntimeError("llama-cpp server failed to start within timeout")

# --- GPU-specific functions with volumes ---
@app.function(
    gpu="T4", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_t4(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_llamacpp_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="L4", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_l4(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_llamacpp_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="A10G", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_a10g(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_llamacpp_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="L40S", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_l40s(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_llamacpp_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="A100", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_a100_40gb(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_llamacpp_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="A100-80GB", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)  
def run_chat_a100_80gb(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_llamacpp_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="H100", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_h100(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_llamacpp_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="H200", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_h200(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_llamacpp_logic(model_name, custom_questions, api_only)

@app.function(
    gpu="B200", 
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def run_chat_b200(model_name: str, custom_questions: Optional[list] = None, api_only: bool = False):
    return run_llamacpp_logic(model_name, custom_questions, api_only)

# --- Helper function ---
def get_chat_function(model_name: str):
    """Get the appropriate chat function based on model requirements"""
    gpu_config = get_gpu_config_llamacpp(model_name)
    if len(gpu_config) >= 2:
        gpu_type = gpu_config[0]
    else:
        gpu_type = "L4"
    
    function_map = {
        "T4": run_chat_t4,
        "L4": run_chat_l4,
        "A10G": run_chat_a10g,
        "L40S": run_chat_l40s,
        "A100-40GB": run_chat_a100_40gb,
        "A100-80GB": run_chat_a100_80gb, 
        "H100": run_chat_h100,
        "H200": run_chat_h200,
        "B200": run_chat_b200,
    }
    
    return function_map.get(gpu_type, run_chat_l4)

# --- Model management with volumes ---
@app.function(
    image=base_image, 
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=3600,
    volumes={"/models": default_volume}
)
def download_model_remote(model_name: str):
    """Download a model to persistent volume"""
    print(f"📥 Downloading model to volume: {model_name}")
    model_path = get_model_path_from_name(model_name)
    download_model_to_path(model_name, model_path)
    return f"✅ Model {model_name} downloaded successfully!"

@app.function(
    image=base_image,
    volumes={"/models": default_volume}
)
def list_model_files():
    """List files in the default GGUF model's volume"""
    model_name = DEFAULT_MODEL
    model_path = get_model_path_from_name(model_name)
    
    if not model_path.exists():
        return f"❌ Model {model_name} not found in volume"
    
    files = list(model_path.glob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    # Check GGUF files
    gguf_files = list(model_path.glob("*.gguf"))
    
    result = [f"📁 Model: {model_name}"]
    result.append(f"📦 Using persistent volume storage")
    result.append(f"📁 Path: {model_path}")
    result.append(f"📊 Total size: {total_size / 1e9:.2f} GB")
    result.append(f"🗜️ GGUF files: {len(gguf_files)}")
    result.append(f"📄 Files ({len(files)}):")
    
    for file in sorted(files):
        if file.is_file():
            size_mb = file.stat().st_size / 1e6
            # Mark GGUF files
            gguf_marker = " 🗜️" if file.name.endswith('.gguf') else ""
            result.append(f"   ✅ {file.name} ({size_mb:.1f} MB){gguf_marker}")
        else:
            result.append(f"   📁 {file.name}/")
    
    return "\n".join(result)

@app.function(
    image=base_image,
    volumes={"/models": default_volume}
)
def delete_model_from_volume(model_name: str):
    """Delete a model from its volume"""
    import shutil
    
    model_path = get_model_path_from_name(model_name)
    
    if not model_path.exists():
        return f"❌ Model {model_name} not found in volume"
    
    print(f"🗑️ Deleting model {model_name} from volume...")
    shutil.rmtree(model_path)
    
    return f"✅ Model {model_name} deleted from volume"

@app.function(
    image=base_image,
    volumes={"/models": default_volume}
)
def cleanup_gguf_model(model_name: str):
    """Clean up GGUF model - keep only Q8_0 and essential files"""
    model_path = get_model_path_from_name(model_name)
    
    if not model_path.exists():
        return f"❌ Model {model_name} not found in volume"
    
    if not is_gguf_model(model_name):
        return f"❌ Model {model_name} is not a GGUF model"
    
    print(f"🧹 Cleaning up GGUF model: {model_name}")
    
    # Get all files
    all_files = list(model_path.glob("*"))
    total_size_before = sum(f.stat().st_size for f in all_files if f.is_file())
    
    # Find the best GGUF file (Q8_0)
    try:
        best_gguf = get_gguf_file_path(model_path)
        print(f"🎯 Keeping best GGUF file: {best_gguf.name}")
    except Exception as e:
        return f"❌ Could not find GGUF file: {e}"
    
    # Files to keep
    keep_patterns = [
        "*.json",          # Config files
        "*.txt",           # README, etc.
        "*.md",            # Documentation
        "tokenizer*",      # Tokenizer files
        best_gguf.name,    # The selected GGUF file
    ]
    
    files_to_keep = set()
    for pattern in keep_patterns:
        files_to_keep.update(model_path.glob(pattern))
    
    # Delete unwanted GGUF files
    deleted_files = []
    deleted_size = 0
    
    for file in all_files:
        if file.is_file() and file not in files_to_keep:
            if file.name.endswith('.gguf'):
                size_gb = file.stat().st_size / 1e9
                print(f"🗑️ Deleting: {file.name} ({size_gb:.1f} GB)")
                deleted_size += file.stat().st_size
                deleted_files.append(file.name)
                file.unlink()
    
    # Calculate savings
    total_size_after = sum(f.stat().st_size for f in model_path.glob("*") if f.is_file())
    saved_gb = (total_size_before - total_size_after) / 1e9
    
    result = [
        f"✅ GGUF cleanup completed for {model_name}",
        f"🎯 Kept: {best_gguf.name}",
        f"🗑️ Deleted {len(deleted_files)} files:",
    ]
    
    for file in deleted_files:
        result.append(f"   - {file}")
    
    result.extend([
        f"💾 Size before: {total_size_before / 1e9:.2f} GB",
        f"💾 Size after: {total_size_after / 1e9:.2f} GB", 
        f"💰 Saved: {saved_gb:.2f} GB ({saved_gb/total_size_before*100:.1f}%)"
    ])
    
    return "\n".join(result)

# --- Local entrypoints ---
@app.local_entrypoint()
def chat(questions: str = ""):
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_config = get_gpu_config_llamacpp(current_model)
    gpu_type = gpu_config[0] if len(gpu_config) >= 1 else "L4"
    
    print(f"🚀 Starting chat session...")
    print(f"🤖 Model: {current_model}")
    print(f"🔧 GPU: {gpu_type}")
    print(f"📦 Using persistent volume for model storage")
    print(f"🦙 Engine: llama-cpp-python v0.2.24")
    print(f"🗜️ GGUF Q8_0 quantization support")
    
    custom_questions = None
    if questions:
        custom_questions = [q.strip() for q in questions.split("|") if q.strip()]
        print(f"📝 Using {len(custom_questions)} custom questions")
    
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions, api_only=False)

@app.local_entrypoint()
def serve_api():
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_config = get_gpu_config_llamacpp(current_model)
    gpu_type = gpu_config[0] if len(gpu_config) >= 1 else "L4"
    
    print(f"🌐 Starting API-only server...")
    print(f"🤖 Model: {current_model}")
    print(f"🔧 GPU: {gpu_type}")
    print(f"📦 Using persistent volume for model storage")
    print(f"🦙 Engine: llama-cpp-python v0.2.24")
    print(f"🗜️ GGUF Q8_0 quantization support")
    
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions=None, api_only=True)

@app.local_entrypoint()
def serve_demo():
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_config = get_gpu_config_llamacpp(current_model)
    gpu_type = gpu_config[0] if len(gpu_config) >= 1 else "L4"
    
    print(f"🧪 Starting API demo server (5 minutes)...")
    print(f"🤖 Model: {current_model}")
    print(f"🔧 GPU: {gpu_type}")
    print(f"📦 Using persistent volume for model storage")
    print(f"🦙 Engine: llama-cpp-python v0.2.24")
    print(f"🗜️ GGUF Q8_0 quantization support")
    
    chat_func = get_chat_function(current_model)
    chat_func.remote(current_model, custom_questions=[], api_only=False)

@app.local_entrypoint()
def test_gguf_q8():
    """Test GGUF Q8_0 quantization with default model"""
    model_name = DEFAULT_MODEL
    print(f"🧪 Testing GGUF Q8_0 quantization: {model_name}")
    print(f"🔧 This will automatically select H100 for 18.4B MoE GGUF")
    print(f"🗜️ Q8_0 format with built-in quantization")
    print(f"🦙 Using llama-cpp-python for optimal GGUF support")
    
    chat_func = get_chat_function(model_name)
    chat_func.remote(model_name, [
        "Hello! Testing GGUF Q8_0 quantization with llama-cpp.",
        "What is a Mixture of Experts model and how does it work?",
        "Thank you for demonstrating Q8_0 quantization!"
    ], api_only=False)

@app.local_entrypoint()
def test_small_gguf():
    """Test with a smaller GGUF model"""
    model_name = "microsoft/Phi-3-mini-4k-instruct-gguf"
    print(f"🧪 Testing small GGUF model: {model_name}")
    print(f"🦙 Using llama-cpp-python for GGUF support")
    
    chat_func = get_chat_function(model_name)
    chat_func.remote(model_name, [
        "Hello! Testing a small GGUF model with llama-cpp.",
        "What are the benefits of GGUF format?",
        "Thank you for the demonstration!"
    ], api_only=False)

@app.local_entrypoint()
def benchmark_llamacpp():
    """Benchmark llama-cpp performance"""
    model_name = DEFAULT_MODEL
    print(f"🏁 Benchmarking llama-cpp with {model_name}")
    print(f"📊 Measuring startup time and inference speed")
    
    start_time = time.time()
    chat_func = get_chat_function(model_name)
    
    # Simple benchmark
    chat_func.remote(model_name, [
        "Count from 1 to 10.",
        "What is 2+2?",
        "Thank you!"
    ], api_only=False)
    
    total_time = time.time() - start_time
    print(f"📊 Total benchmark time: {total_time:.1f}s")

@app.local_entrypoint()
def gpu_specs():
    """Show GPU specifications and recommended GGUF models"""
    print("🚀 GPU Specifications & GGUF Model Recommendations")
    print("🦙 llama-cpp-python with Q8_0 Quantization Support!")
    print("=" * 80)
    
    specs = [
        ("T4", "16GB", "1-3B GGUF", "microsoft/Phi-3-mini-4k-instruct-gguf"),
        ("L4", "24GB", "3-7B GGUF", "bartowski/Qwen2.5-7B-Instruct-GGUF"),
        ("A10G", "24GB", "3-7B GGUF", "bartowski/Llama-3.2-3B-Instruct-GGUF"),
        ("L40S", "48GB", "7-13B GGUF", "bartowski/Qwen2.5-14B-Instruct-GGUF"),
        ("A100-80GB", "80GB", "13-27B GGUF", "bartowski/Qwen2.5-32B-Instruct-GGUF"),
        ("H100", "80GB", "18B MoE GGUF", DEFAULT_MODEL),
        ("H200", "141GB", "34B+ GGUF", "bartowski/Yi-1.5-34B-Chat-GGUF"),
        ("B200", "192GB", "70B+ GGUF", "bartowski/Llama-3.1-70B-Instruct-GGUF"),
    ]
    
    print("\n🔧 GPU | Memory | Model Size | Recommended GGUF Example")
    print("-" * 80)
    for gpu, memory, models, example in specs:
        print(f"{gpu:8} | {memory:7} | {models:15} | {example}")
    
    print(f"\n🦙 llama-cpp Advantages:")
    print(f"  ✅ Native Q8_0 quantization support")
    print(f"  ✅ Faster startup time (< 30s)")
    print(f"  ✅ Lower VRAM usage")
    print(f"  ✅ Better GGUF compatibility")
    print(f"  ✅ MoE model optimizations")
    print(f"  ✅ Built-in OpenAI API compatibility")
    
    print(f"\n💬 GGUF Examples:")
    print(f"  Small:      MODEL_NAME='microsoft/Phi-3-mini-4k-instruct-gguf' modal run vllmserver.py::serve_api")
    print(f"  Medium:     MODEL_NAME='bartowski/Qwen2.5-7B-Instruct-GGUF' modal run vllmserver.py::serve_api")
    print(f"  Large MoE:  MODEL_NAME='{DEFAULT_MODEL}' modal run vllmserver.py::serve_api")
    print(f"  Test Q8:    modal run vllmserver.py::test_gguf_q8")

@app.local_entrypoint()
def download(model_name: str = None):
    """Download a GGUF model to persistent volume"""
    if not model_name:
        model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    
    if not is_gguf_model(model_name):
        print("❌ Only GGUF models are supported in llama-cpp mode")
        print("💡 Try one of these GGUF models:")
        print("   - microsoft/Phi-3-mini-4k-instruct-gguf")
        print("   - bartowski/Qwen2.5-7B-Instruct-GGUF")
        print(f"   - {DEFAULT_MODEL}")
        return
    
    print(f"📥 Downloading GGUF model to persistent volume: {model_name}")
    print(f"🎯 Will prioritize Q8_0 quantization")
    print(f"🦙 Optimized for llama-cpp-python")
    result = download_model_remote.remote(model_name)
    print(result)

@app.local_entrypoint()
def list_files(model_name: str = None):
    """List files in a model's volume"""
    if not model_name:
        model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    
    result = list_model_files.remote()
    print(result)

@app.local_entrypoint()
def delete_model(model_name: str = None):
    """Delete a model from volume"""
    if not model_name:
        model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    
    print(f"⚠️  Are you sure you want to delete {model_name}? This cannot be undone.")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm.lower() == 'yes':
        result = delete_model_from_volume.remote(model_name)
        print(result)
    else:
        print("❌ Deletion cancelled")

@app.local_entrypoint()
def cleanup_gguf():
    """Clean up GGUF model - remove unwanted quantizations"""
    model_name = DEFAULT_MODEL
    print(f"🧹 Cleaning up GGUF model: {model_name}")
    print(f"🎯 Will keep only Q8_0 (best quality) and essential files")
    print(f"🗑️ Will delete other quantization levels to save space")
    
    result = cleanup_gguf_model.remote(model_name)
    print(result)

@app.local_entrypoint()
def info():
    current_model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu_config = get_gpu_config_llamacpp(current_model)
    gpu_type = gpu_config[0] if len(gpu_config) >= 1 else "L4"
    memory_util = gpu_config[1] if len(gpu_config) >= 2 else 0.85
    
    print(f"🦙 llama-cpp-python v0.2.24 with GGUF Q8_0 Support:")
    print(f"  Model: {current_model}")
    print(f"  GPU: {gpu_type} ({memory_util*100}% memory)")
    print(f"  Engine: llama-cpp-python (replaces vLLM)")
    print(f"  Quantization: GGUF built-in (Q8_0 preferred)")
    print(f"  Volume: llamacpp-models-storage")
    
    print(f"\n🔧 Migration Completed:")
    print(f"  ✅ vLLM completely replaced with llama-cpp")
    print(f"  ✅ Native Q8_0 quantization support")
    print(f"  ✅ Faster startup time")
    print(f"  ✅ Lower VRAM usage")
    print(f"  ✅ Better GGUF compatibility")
    print(f"  ✅ MoE model optimizations")
    print(f"  ✅ OpenAI API compatibility maintained")
    
    print(f"\n📋 Available Commands:")
    print(f"  serve_api               - Run API server indefinitely")
    print(f"  test_gguf_q8           - Test Q8_0 quantization")
    print(f"  test_small_gguf        - Test small GGUF model")
    print(f"  benchmark_llamacpp     - Performance benchmark")
    print(f"  gpu_specs              - Show GGUF recommendations")
    print(f"  download                - Download GGUF model")
    print(f"  cleanup_gguf           - Clean up quantizations")
    print(f"  info                    - Show this info")
