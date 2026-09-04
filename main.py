"""
Foundry Local ile Local RAG (Retrieval-Augmented Generation) Uygulaması
------------------------------------------------------------------------
Microsoft Foundry Local SDK, ONNX Runtime ve SQLite Veritabanı kullanılarak
geliştirilmiş çevrimdışı (offline) RAG mimarisi.
"""

import math
import sqlite3
import json
from foundry_local_sdk import Configuration, FoundryLocalManager

# ---------------------------------------------------------------------------
# 1) BİLGİ TABANI (KNOWLEDGE BASE - SANA ÖZEL İÇERİK)
# ---------------------------------------------------------------------------
documents = [
    "Computer architecture defines the functionality, organization, and implementation of computer systems.",
    "MIPS is a Reduced Instruction Set Computer (RISC) architecture that uses fixed-length 32-bit instructions.",
    "ARMv7 is a 32-bit RISC architecture widely used in embedded systems and system-on-chip platforms like DE1-SoC.",
    "The MARS simulator is a lightweight interactive development environment for programming in MIPS assembly language.",
    "In C++ and computer systems, pointers store memory addresses and enable direct memory manipulation.",
    "Data structures like arrays, linked lists, and binary search trees organize and store data for efficient access.",
    "Time complexity analysis using Big-O notation measures the execution efficiency and scalability of algorithms.",
    "Signals and systems engineering focuses on mathematical modeling, continuous and discrete signals, and convolution operations.",
    "Convolution in linear time-invariant systems computes the output response by combining an input signal with an impulse response.",
    "Foundry Local enables developers to run machine learning models and local LLMs offline on client devices using ONNX Runtime.",
]


# ---------------------------------------------------------------------------
# 2) SQLITE VERİTABANI VE RETRIEVAL FONKSİYONLARI
# ---------------------------------------------------------------------------
def init_db():
    """SQLite veritabanını başlatır ve tabloyu hazırlar."""
    conn = sqlite3.connect("knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            embedding TEXT
        )
    """)
    conn.commit()
    conn.close()


def store_documents(embedding_client, doc_list):
    """Dokümanları vektörleştirip SQLite veritabanına kaydeder."""
    conn = sqlite3.connect("knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents")  # Temiz başlangıç
    
    response = embedding_client.generate_embeddings(doc_list)
    for doc_text, item in zip(doc_list, response.data):
        emb_json = json.dumps(item.embedding)
        cursor.execute("INSERT INTO documents (content, embedding) VALUES (?, ?)", (doc_text, emb_json))
    
    conn.commit()
    conn.close()


def cosine_similarity(a, b):
    """İki vektör arasındaki kosinüs benzerliğini hesaplar."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def search_similar_documents(query_embedding, top_k=2):
    """SQLite'taki vektörler arasında kosinüs benzerliği araması yapar."""
    conn = sqlite3.connect("knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("SELECT content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    scores = []
    for content, emb_str in rows:
        emb_vector = json.loads(emb_str)
        score = cosine_similarity(query_embedding, emb_vector)
        scores.append((content, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ---------------------------------------------------------------------------
# 3) ANA PROGRAM
# ---------------------------------------------------------------------------
def main():
    # SQLite Veritabanını Kur
    init_db()

    # Foundry Local SDK Başlat
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # 1. Embedding Modelini Yükle
    print("\n[1/3] Embedding modeli hazırlanıyor...")
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rEmbedding modeli indiriliyor: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # 2. Dokümanları SQLite'a İndeksle
    print("[2/3] Bilgi tabanı SQLite veritabanına indeksleniyor...")
    store_documents(embedding_client, documents)
    print(f"{len(documents)} doküman SQLite 'knowledge_base.db' içine kaydedildi.")

    # 3. Chat Modelini Yükle
    print("\n[3/3] Chat modeli hazırlanıyor...")
    chat_model = manager.catalog.get_model("qwen2.5-0.5b")
    chat_model.download(
        lambda p: print(f"\rChat modeli indiriliyor: {p:.1f}%", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\n" + "="*50)
    print("SİSTEM HAZIR: Foundry Local & SQLite RAG Asistanı")
    print("="*50)
    print("Örnek soru: What is MIPS architecture?")
    print("Çıkmak için: quit\n")

    # Soru - Cevap Döngüsü
    while True:
        query = input("Soru Sorun: ").strip()
        if not query or query.lower() == "quit":
            break

        # Sorguyu vektörleştir
        query_response = embedding_client.generate_embedding(query)
        query_embedding = query_response.data[0].embedding

        # SQLite üzerinden en alakalı belgeleri getir
        results = search_similar_documents(query_embedding, top_k=2)
        context = "\n".join(f"- {content}" for content, _ in results)

        # Prompt oluştur
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful local technical assistant. "
                    "Answer the user's question using ONLY the provided context. "
                    "If the context doesn't contain the answer, state that clearly.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": query},
        ]

        # Güvenli Yanıt Akışı (Stream Fix)
        print("\nCevap: ", end="", flush=True)
        for chunk in chat_client.complete_streaming_chat(messages):
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    print(content, end="", flush=True)
        print("\n" + "-"*40 + "\n")

    # Belleği Temizle
    embedding_model.unload()
    chat_model.unload()
    print("Oturum kapatıldı. Modeller bellekten kaldırıldı.")


if __name__ == "__main__":
    main()