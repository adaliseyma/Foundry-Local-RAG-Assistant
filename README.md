# Building Your First Local RAG Application with Foundry Local & SQLite

Bu proje, Microsoft Foundry Local SDK ve ONNX Runtime kullanılarak geliştirilmiş, tamamen yerel (offline) çalışan bir **Retrieval-Augmented Generation (RAG)** soru-cevap uygulamasıdır. 

Uygulama, harici bir bulut API'sine veya internet bağlantısına ihtiyaç duymadan cihaz üzerinde çalışan yerel dil modelleri (SLM/LLM) ve SQLite vektör depolaması ile teknik soruları yanıtlar.

---

## 🏗️ Sistem Mimarisi & Çalışma Mantığı

1. **İndeksleme (Data Ingestion):** Bilgi tabanındaki teknik metinler `qwen3-embedding-0.6b` modeli ile yerel olarak vektörleştirilir ve `knowledge_base.db` (SQLite) içine kaydedilir.
2. **Benzerlik Araması (Retrieval):** Kullanıcı bir soru sorduğunda sorgu vektöre dönüştürülür ve SQLite veritabanındaki kayıtlarla **Cosine Similarity** (Kosinüs Benzerliği) hesaplanarak en alakalı bağlam (Top-K context) çekilir.
3. **Yanıt Üretimi (Generation):** Yakalanan bağlam ve kullanıcı sorusu, `qwen2.5-0.5b` yerel sohbet modeline iletilerek gerçek zamanlı akış (streaming) ile cevap üretilir.

---

## 🛠️ Kullanılan Teknolojiler

- **Çalışma Ortamı:** Python 3.11+
- **Yerel AI Motoru:** Microsoft Foundry Local SDK & ONNX Runtime
- **Embedding Modeli:** `qwen3-embedding-0.6b`
- **Sohbet Modeli:** `qwen2.5-0.5b`
- **Veritabanı:** SQLite3
- **Platform Desteği:** Windows (WinML) / macOS / Linux

---

## 🚀 Kurulum & Çalıştırma

### 1. Gereksinimleri Yükleyin
```bash
# Windows için
pip install -r requirements.txt

# veya doğrudan
pip install foundry-local-sdk-winml openai