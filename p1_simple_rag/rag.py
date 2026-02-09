import os
import re
import json
import requests
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from pypdf import PdfReader
import docx

DATA_DIR = "./data"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gpt-oss:20b"

# ---------- Loaders ----------
def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def load_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

def load_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt" or ext == ".md":
        return load_txt(path)
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")

# ---------- Chunking ----------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120):
    """
    Simple char-based chunking.
    chunk_size ~ 900 chars keeps context manageable and fast.
    """
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

# ---------- Vector DB ----------
def get_chroma_collection():
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(name=COLLECTION_NAME)

# ---------- Indexing ----------
def index_documents():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Created {DATA_DIR}. Put docs there and re-run.")
        return

    files = []
    for root, _, filenames in os.walk(DATA_DIR):
        for fn in filenames:
            if fn.lower().endswith((".txt", ".md", ".pdf", ".docx")):
                files.append(os.path.join(root, fn))

    if not files:
        print(f"No supported files found in {DATA_DIR} (.txt .md .pdf .docx).")
        return

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    col = get_chroma_collection()

    # Avoid duplicate re-indexing: use deterministic IDs
    ids = []
    metadatas = []
    documents = []

    for path in tqdm(files, desc="Loading+Chunking"):
        try:
            raw = load_file(path)
        except Exception as e:
            print(f"Skip {path}: {e}")
            continue

        chunks = chunk_text(raw)
        base = os.path.relpath(path, DATA_DIR)

        for i, ch in enumerate(chunks):
            doc_id = f"{base}::chunk_{i}"
            ids.append(doc_id)
            metadatas.append({"source": base, "chunk": i})
            documents.append(ch)

    # Filter out already existing IDs
    existing = set()
    try:
        # Chroma doesn't have a simple "exists" call; we query in batches
        # We'll just attempt add; duplicates may error depending on version.
        pass
    except Exception:
        pass

    print(f"Embedding {len(documents)} chunks...")
    embeddings = embedder.encode(documents, show_progress_bar=True, normalize_embeddings=True).tolist()

    # Add to Chroma
    col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print("✅ Indexing complete.")

# ---------- Retrieval ----------
def retrieve(query: str, k: int = 4):
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()[0]

    col = get_chroma_collection()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    hits = []
    for d, m, dist in zip(docs, metas, dists):
        hits.append({"text": d, "meta": m, "distance": dist})
    return hits

# ---------- Generation (Ollama) ----------
def ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "")

def build_prompt(question: str, contexts):
    context_block = "\n\n".join(
        [f"[Source: {c['meta']['source']} | chunk {c['meta']['chunk']}]\n{c['text']}" for c in contexts]
    )
    return f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the provided documents."

Context:
{context_block}

Question: {question}

Answer:"""

# ---------- CLI ----------
def chat():
    print("\nRAG chat. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        hits = retrieve(q, k=4)
        prompt = build_prompt(q, hits)
        ans = ollama_generate(prompt)

        print("\nAssistant:", ans.strip(), "\n")
        print("Top sources:")
        for h in hits:
            print(f" - {h['meta']['source']} (chunk {h['meta']['chunk']}, distance {h['distance']:.4f})")
        print()

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--index", action="store_true", help="Index documents from ./data")
    p.add_argument("--chat", action="store_true", help="Start chat")
    args = p.parse_args()

    if args.index:
        index_documents()
    if args.chat:
        chat()
    if not args.index and not args.chat:
        p.print_help()

if __name__ == "__main__":
    main()
