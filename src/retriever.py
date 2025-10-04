import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "chunks/chunks.json"          
EMB_FILE = "vectordb/embeddings.npy"       
FAISS_FILE = "vectordb/faiss.index"        


def load_corpus():
    if not os.path.exists(CHUNK_FILE):
        raise FileNotFoundError(f" Missing {CHUNK_FILE}. Run ingest.py first.")
    if not os.path.exists(EMB_FILE) or not os.path.exists(FAISS_FILE):
        raise FileNotFoundError(" Missing embeddings/index in vectordb/. Run ingest.py first.")

    with open(CHUNK_FILE, "r") as f:
        chunks = json.load(f)
    embs = np.load(EMB_FILE)
    index = faiss.read_index(FAISS_FILE)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return chunks, embs, index, model


def retrieve(query, chunks, index, model, top_k=3):
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_vec, top_k)
    return [chunks[i] for i in I[0]]
