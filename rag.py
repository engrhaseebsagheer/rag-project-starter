"""
rag.py - a minimal RAG (retrieval-augmented generation) starter in one file.

Ask questions about a PDF and get answers that cite the page they came from.
Uses the free tier of the Gemini API. No vector database: vectors live in memory.

    pip install -r requirements.txt
    export GEMINI_API_KEY=your_key        # Windows: set GEMINI_API_KEY=your_key
    python rag.py notes.pdf "What is a deadlock?"
    python rag.py notes.pdf               # interactive mode

Full guide: https://www.kitsbyai.com/how-to-build-a-rag-app
"""
import os
import sys

import numpy as np
import requests
from pypdf import PdfReader

API = "https://generativelanguage.googleapis.com/v1beta"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
DIM = 768            # shorter vectors: smaller and faster, almost no quality loss
CHUNK_SIZE = 1000    # characters per chunk
OVERLAP = 150        # characters shared by neighbouring chunks, so sentences are not cut in half
TOP_K = 6            # how many chunks the model gets to read


def post(path, body):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY first. Get a free key at https://aistudio.google.com/apikey")
    r = requests.post(f"{API}/{path}", json=body, headers={"x-goog-api-key": key}, timeout=60)
    if r.status_code != 200:
        sys.exit(f"Gemini API error {r.status_code}: {r.text[:300]}")
    return r.json()


# 1. Read the PDF page by page, keeping the page number with every chunk.
def load_chunks(path):
    chunks = []
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        start = 0
        while start < len(text):
            piece = text[start:start + CHUNK_SIZE]
            if len(piece) > 50:
                chunks.append({"page": page_no, "text": piece})
            start += CHUNK_SIZE - OVERLAP
    if not chunks:
        sys.exit("No text found. Scanned PDFs (photos of pages) need OCR first.")
    return chunks


# 2. Turn text into vectors. Documents and questions use different task types.
def embed(texts, task):
    vectors = []
    for i in range(0, len(texts), 50):  # the batch endpoint takes up to 100 texts
        body = {"requests": [{
            "model": f"models/{EMBED_MODEL}",
            "content": {"parts": [{"text": t}]},
            "taskType": task,
            "outputDimensionality": DIM,
        } for t in texts[i:i + 50]]}
        vectors += [e["values"] for e in post(f"models/{EMBED_MODEL}:batchEmbedContents", body)["embeddings"]]
    v = np.array(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # unit length, so dot product = cosine similarity


# 3. Find the chunks closest in meaning to the question.
def search(question, chunks, matrix):
    q = embed([question], "RETRIEVAL_QUERY")[0]
    scores = matrix @ q
    best = np.argsort(-scores)[:TOP_K]
    return [chunks[i] | {"score": float(scores[i])} for i in best]


# 4. Ask the model to answer only from those chunks, citing them as [1], [2].
def answer(question, hits):
    sources = "\n\n".join(f"[{n}] (page {h['page']})\n{h['text']}" for n, h in enumerate(hits, start=1))
    prompt = (
        "Answer the question using only the numbered sources below. "
        "Cite sources like [1] or [2] after each fact. "
        "If the sources do not contain the answer, say you could not find it in the document.\n\n"
        f"Sources:\n{sources}\n\nQuestion: {question}"
    )
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2}}
    data = post(f"models/{CHAT_MODEL}:generateContent", body)
    return data["candidates"][0]["content"]["parts"][0]["text"]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    chunks = load_chunks(sys.argv[1])
    print(f"Indexing {len(chunks)} chunks from {sys.argv[1]} ...")
    matrix = embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    questions = [sys.argv[2]] if len(sys.argv) > 2 else None
    while True:
        q = questions.pop() if questions else input("\nAsk (or press Enter to quit): ").strip()
        if not q:
            break
        hits = search(q, chunks, matrix)
        print("\n" + answer(q, hits))
        print("\nSources: " + ", ".join(f"[{n}] page {h['page']}" for n, h in enumerate(hits, start=1)))
        if questions is not None and not questions:
            break


if __name__ == "__main__":
    main()
