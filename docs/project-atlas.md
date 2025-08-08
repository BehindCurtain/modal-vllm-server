# Modal llama-cpp Server - Proje Atlası

## Proje Amacı

Modal llama-cpp Server, serverless GPU altyapısı üzerinde çalışan, OpenAI uyumlu GGUF model API servisidir. Kullanıcıların herhangi bir GPU konfigürasyonu yapmadan, otomatik olarak optimize edilmiş şekilde GGUF quantized modellerini servis etmelerini sağlar.

## Temel Felsefe

- **GGUF Optimizasyonu**: Q8_0 quantization ile optimal performans
- **Kolay Kullanım**: Tek komutla GGUF model servisi başlatma
- **Maliyet Etkinliği**: Modal'ın serverless yapısı ile sadece kullanıldığında ödeme
- **Uyumluluk**: OpenAI API standardına tam uyum
- **Hızlı Başlatma**: vLLM'den 3x daha hızlı startup süresi

## Mimari Genel Bakış

```
┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐
│ 👤 Kullanıcı    │───▶│ 🎯 GPU Seçimi    │───▶│ 🦙 llama-cpp  │
│ İsteği          │    │   Algoritması     │    │   Server      │
└─────────────────┘    └──────────────────┘    └───────────────┘
                              │                        │
                              ▼                        ▼
                       ┌──────────────────┐    ┌───────────────┐
                       │ 📥 GGUF Model    │    │ 🔌 OpenAI API │
                       │ Yönetimi         │    │ Endpoint      │
                       └──────────────────┘    └───────────────┘
```

## Teknoloji Stack

- **Modal**: Serverless GPU infrastructure
- **llama-cpp-python**: High-performance GGUF inference engine
- **FastAPI**: Modern web framework (llama-cpp içinde)
- **Hugging Face**: Model repository ve tokenizer
- **CUDA**: GPU acceleration with CUBLAS

## Desteklenen Model Formatları

- **SafeTensors**: Güvenli tensor formatı (önerilen)
- **PyTorch**: Geleneksel .bin formatı
- **GGUF**: Quantized model formatı (v0.10.0+) ✅ YENİ!
- **Quantization**: GPTQ, AWQ, BitsAndBytes

### GGUF Format Desteği (Yeni Özellik)
- **Tek dosyalı yapı**: Tüm model tek .gguf dosyasında
- **Built-in quantization**: Dahili sıkıştırma
- **MoE model desteği**: Mixture of Experts modeller
- **Otomatik GPU seçimi**: 18.4B MoE için H100 seçimi
- **Hedef model**: `DavidAU/Llama-3.2-8X3B-MOE-Dark-Champion-Instruct-uncensored-abliterated-18.4B-GGUF`

## GPU Seçimi Stratejisi

| Model Boyutu | Quantized | GPU Seçimi | Memory Kullanımı |
|-------------|-----------|------------|------------------|
| 70B+        | ❌        | B200/H200  | 95% |
| 70B         | ✅        | H100       | 85% |
| 30B-65B     | ❌        | H100       | 85% |
| 13B-30B     | ❌        | A100-80GB  | 80% |
| 7B-13B      | ❌        | L40S       | 75% |
| 3B-7B       | Any       | A10G       | 75% |
| 1B-3B       | Any       | L4         | 70% |

## Temel Prensipler

1. **Memory Safety**: Konservatif memory allocation ile OOM hatalarını önleme
2. **Auto-scaling**: Modal'ın otomatik ölçeklendirme özelliklerini kullanma
3. **Persistent Storage**: Model cache için volume kullanımı
4. **Error Resilience**: Kapsamlı hata yönetimi ve fallback mekanizmaları
5. **Performance**: Model boyutuna göre optimize edilmiş konfigürasyonlar

## Kullanım Senaryoları

- **Geliştirme**: Hızlı prototipleme ve test
- **Prodüksiyon**: Ölçeklenebilir API servisi
- **Araştırma**: Farklı modelleri deneme
- **RP/Chat**: Roleplay ve sohbet uygulamaları
