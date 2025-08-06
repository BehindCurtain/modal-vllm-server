# Model Yönetim Sistemi

## Genel Bakış

Model Yönetim Sistemi, model indirme, önbellekleme, format tespiti ve konfigürasyon işlemlerini yöneten sistemdir.

## Ana Fonksiyonlar

### `get_model_path_from_name(model_name: str) -> Path`

Model adından güvenli dosya yolu oluşturur.

**İşlem:**
1. Model adındaki `/` karakterlerini `--` ile değiştirir
2. Güvenli dizin adı oluşturur
3. `/models` volume'u altında path döner

### `download_model_to_path(model_name: str, model_path: Path)`

Model indirme ve integrity check işlemleri.

**Özellikler:**
- Resume interrupted downloads
- Integrity checking
- Sharded model detection
- Automatic retry on corruption

### `get_model_config(model_path: Path) -> dict`

Model konfigürasyon dosyasını okur ve anahtar parametreleri çıkarır.

**Çıkarılan Parametreler:**
- `max_position_embeddings`
- `model_max_length`
- `vocab_size`
- `hidden_size`
- `model_type`
- `architectures`

## Model Format Desteği

### SafeTensors Format
- **Dosyalar**: `model.safetensors` veya `model-*.safetensors`
- **Index**: `model.safetensors.index.json`
- **Avantajlar**: Güvenli, hızlı yükleme
- **Durum**: Tam destek

### PyTorch Format
- **Dosyalar**: `pytorch_model.bin` veya `pytorch_model-*.bin`
- **Index**: `pytorch_model.bin.index.json`
- **Avantajlar**: Yaygın kullanım
- **Durum**: Tam destek

### GGUF Format (Yeni)
- **Dosyalar**: `*.gguf` (tek dosya)
- **Avantajlar**: Quantized, küçük boyut
- **Gereksinimler**: vLLM 0.10.0+
- **Durum**: Ekleniyor

## Sharding Detection

### `check_model_sharding(model_path: Path) -> tuple[bool, int, list]`

Model'in sharded olup olmadığını tespit eder.

**Kontrol Edilen Dosyalar:**
1. `pytorch_model.bin.index.json`
2. `model.safetensors.index.json`
3. `pytorch_model-*.bin` pattern
4. `model-*.safetensors` pattern

**Dönen Değerler:**
- `is_sharded`: Boolean
- `shard_count`: Shard sayısı
- `shard_files`: Shard dosya listesi

## Chat Template Yönetimi

### `setup_chat_template(model_path: Path, model_name: str)`

Model için uygun chat template'i konfigüre eder.

**Desteklenen Model Tipleri:**
- **Yi modeller**: `<|im_start|>` format
- **Qwen modeller**: `<|im_start|>` format
- **Mistral modeller**: `[INST]` format
- **DialoGPT**: Basit format
- **Generic**: `### Human/Assistant` format

**RP Uyumluluğu:**
- Roleplay senaryoları için optimize
- Character consistency
- Context preservation

## Model Compatibility

### `is_model_supported(model_config: dict, model_name: str) -> bool`

Model'in vLLM ile uyumluluğunu kontrol eder.

**Problematik Model Tipleri:**
- `stablelm` (eski versiyonlar)
- `stablelmepoch`
- Bazı custom architectures

**Kontrol Kriterleri:**
- Model type validation
- Architecture compatibility
- vLLM version requirements

## Volume Yönetimi

### Persistent Storage
- **Volume**: `vllm-models-storage`
- **Mount Point**: `/models`
- **Auto-create**: True
- **Sharing**: Tüm fonksiyonlar arası

### Cache Stratejisi
- Model dosyaları kalıcı olarak saklanır
- Tekrar indirme gerekmez
- Disk alanı optimizasyonu
- Automatic cleanup (gelecek özellik)

## Model Lifecycle

### Download Phase
1. Model existence check
2. Integrity validation
3. Download with resume
4. Sharding detection
5. Chat template setup

### Loading Phase
1. Configuration reading
2. Compatibility check
3. Memory estimation
4. vLLM parameter calculation

### Runtime Phase
1. Model serving
2. Performance monitoring
3. Error handling
4. Graceful shutdown

## GGUF Desteği (Yeni Özellik)

### Format Özellikleri
- Tek dosyalı yapı
- Built-in quantization
- Metadata embedded
- Fast loading

### vLLM Integration
- vLLM 0.10.0+ gerekli
- Özel parameter set
- Memory optimization
- Performance tuning

### Detection Logic
```python
def is_gguf_model(model_name: str) -> bool:
    return "gguf" in model_name.lower()

def get_gguf_file(model_path: Path) -> Path:
    gguf_files = list(model_path.glob("*.gguf"))
    if len(gguf_files) == 1:
        return gguf_files[0]
    else:
        raise ValueError("GGUF model must have exactly one .gguf file")
```

## Error Handling

### Download Errors
- Network timeout retry
- Partial download resume
- Corruption detection
- Disk space check

### Loading Errors
- Format mismatch detection
- Missing file handling
- Permission issues
- Memory estimation errors

### Runtime Errors
- Model compatibility issues
- Memory overflow
- Configuration conflicts

## Performance Optimizations

### Download Optimizations
- `HF_HUB_ENABLE_HF_TRANSFER=1`
- Parallel chunk download
- Resume capability
- Bandwidth optimization

### Storage Optimizations
- Deduplication (gelecek)
- Compression (gelecek)
- Automatic cleanup (gelecek)
- Usage tracking (gelecek)

## Monitoring ve Debugging

### Model Information
- File size tracking
- Download progress
- Integrity status
- Usage statistics

### Debug Tools
- `list_model_files()`: Model dosya listesi
- `delete_model_from_volume()`: Model silme
- File integrity check
- Configuration validation

## Gelecek Geliştirmeler

- Multi-format automatic conversion
- Smart caching strategies
- Model versioning
- Automatic updates
- Usage analytics
- Cost optimization
