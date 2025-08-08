# GPU Yönetim Sistemi (llama-cpp)

## Genel Bakış

GPU Yönetim Sistemi, GGUF model boyutu ve tipine göre otomatik olarak en uygun GPU'yu seçen ve llama-cpp için memory kullanımını optimize eden sistemdir. vLLM'den llama-cpp'ye geçiş ile birlikte daha agresif memory utilization ve MoE optimizasyonları eklendi.

## Ana Fonksiyonlar

### `get_gpu_config_llamacpp(model_name: str) -> tuple[str, float, str]`

GGUF model adından GPU tipini, memory utilization değerini ve MoE optimizasyonlarını belirler.

**Algoritma:**
1. GGUF model tespiti (zorunlu)
2. Model adından boyut tespiti (1B, 7B, 70B vb.)
3. MoE model tespiti (8x3B, 8x7B vb.)
4. GPU seçimi tablosuna göre mapping
5. llama-cpp için agresif memory utilization hesaplama
6. MoE optimizasyon parametreleri

**Desteklenen GPU Tipleri:**
- T4 (16GB) - Küçük modeller
- L4 (24GB) - Orta modeller  
- A10G (24GB) - Orta modeller
- L40S (48GB) - Büyük modeller
- A100-40GB/80GB - Çok büyük modeller
- H100 (80GB) - En büyük modeller
- H200 (141GB) - Massive modeller
- B200 (192GB) - Ultra massive modeller

## GPU Seçimi Kuralları

### Model Boyutu Tespiti
```python
# Regex pattern'ler ile model adından boyut çıkarma
if any(size in model_lower for size in ["70b", "72b", "405b"]):
    # Massive model logic
elif any(size in model_lower for size in ["13b", "14b", "17b", "27b", "34b"]):
    # Large model logic
```

### GGUF Model Seçimi (Zorunlu)
```python
if not is_gguf_model(model_name):
    raise ValueError("Only GGUF models are supported in llama-cpp mode")

# MoE model detection
if "moe" in model_lower or "8x3b" in model_lower:
    if "18b" in model_lower or "18.4b" in model_lower:
        return "H100", 0.95, "--n-cpu-moe 80"  # 18.4B MoE GGUF
```

### Memory Utilization Hesaplama (llama-cpp Optimized)

**Faktörler:**
- GGUF built-in quantization
- llama-cpp efficient memory management
- MoE expert routing
- GPU memory kapasitesi

**Agresif Yaklaşım (llama-cpp):**
- GGUF modeller için yüksek utilization (0.85-0.95)
- MoE modeller için CPU offload ile hibrit yaklaşım
- Daha az memory overhead

## Modal Fonksiyon Mapping

Her GPU tipi için ayrı Modal fonksiyonu:

```python
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
```

## Özel Durumlar

### MoE (Mixture of Experts) Modeller
- Daha fazla memory gerektirir
- Özel GPU seçimi kuralları
- Expert routing overhead'i

### GGUF Modeller
- Quantized format
- Tek dosyalı olmalı
- Özel memory hesaplama

### Sharded Modeller
- Çoklu dosya yapısı
- Konservatif memory settings
- Yavaş yükleme süresi

## Hata Yönetimi

### OOM (Out of Memory) Hatası
- Otomatik memory reduction
- Fallback GPU önerisi
- Kullanıcıya alternatif model önerisi

### GPU Uyumsuzluğu
- Model-GPU compatibility check
- Minimum requirements validation
- Graceful degradation

## Performance Optimizasyonları

### llama-cpp Memory Optimizasyonları
- GGUF memory mapping (use_mmap=True)
- CUDA FP16 acceleration
- GPU layer offloading (n_gpu_layers=-1)
- MoE CPU offloading (--n-cpu-moe)

### Environment Variables (llama-cpp)
```python
env["CUDA_VISIBLE_DEVICES"] = "0"
env["GGML_CUDA_ENABLE"] = "1"
env["GGML_CUDA_F16"] = "1"  # FP16 for better performance
```

### llama-cpp Server Configuration
```python
config = {
    "n_ctx": 2048,           # Context length
    "n_batch": 512,          # Batch size
    "n_gpu_layers": -1,      # All layers to GPU
    "use_mmap": True,        # Memory mapping
    "use_mlock": False,      # Memory locking (False for Modal)
    "chat_format": "chatml", # RP-friendly format
}
```

## İzleme ve Debugging

### Startup Diagnostics
- GPU memory check
- Model compatibility validation
- Configuration summary

### Runtime Monitoring
- Memory usage tracking
- Performance metrics
- Error reporting

## Gelecek Geliştirmeler

- Dynamic GPU switching
- Multi-GPU support optimization
- Cost-based GPU selection
- Real-time performance monitoring
