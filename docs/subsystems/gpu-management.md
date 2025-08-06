# GPU Yönetim Sistemi

## Genel Bakış

GPU Yönetim Sistemi, model boyutu ve tipine göre otomatik olarak en uygun GPU'yu seçen ve memory kullanımını optimize eden sistemdir.

## Ana Fonksiyonlar

### `get_gpu_config(model_name: str) -> tuple[str, float]`

Model adından GPU tipini ve memory utilization değerini belirler.

**Algoritma:**
1. Model adından boyut tespiti (1B, 7B, 70B vb.)
2. Quantization tespiti (GPTQ, AWQ, BnB, GGUF)
3. GPU seçimi tablosuna göre mapping
4. Memory utilization hesaplama

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

### Quantization Tespiti
```python
if "gptq" in model_lower:
    quantization = "gptq"
elif "awq" in model_lower:
    quantization = "awq"
elif "gguf" in model_lower:
    quantization = "gguf"
```

### Memory Utilization Hesaplama

**Faktörler:**
- Model boyutu
- Quantization durumu
- GPU memory kapasitesi
- Sharding durumu

**Konservatif Yaklaşım:**
- Büyük modeller için düşük utilization (0.60-0.75)
- Sharded modeller için ekstra düşük (0.55)
- OOM hatalarını önlemek için safety margin

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

### Memory Optimizasyonları
- FP8 KV cache (H100+ için)
- Chunked prefill
- Prefix caching
- Block size optimization

### Environment Variables
```python
env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
env["VLLM_USE_TRITON_FLASH_ATTN"] = "1"  # H100+ için
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
