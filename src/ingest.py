import os, json, re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


PDF_PATH = "C:/Users/HP/OneDrive/PARUL/AI_Chatbot/data/AI Training Document.pdf"
CHUNK_FILE = "chunks/chunks.json"
EMB_FILE = "vectordb/embeddings.npy"
FAISS_FILE = "vectordb/faiss.index"


def read_pdf(path):
    reader = PdfReader(path)
    text = " ".join([p.extract_text() or "" for p in reader.pages])
    return re.sub(r"\s+", " ", text)


def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


def build_embeddings(chunks):
    os.makedirs("chunks", exist_ok=True)
    os.makedirs("vectordb", exist_ok=True)

    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(
        chunks, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    
    with open(CHUNK_FILE, "w") as f:
        json.dump(chunks, f)

    
    np.save(EMB_FILE, embeddings)

    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_FILE)

    print(f"Saved {len(chunks)} chunks → {CHUNK_FILE}")
    print(f"Saved embeddings → {EMB_FILE}")
    print(f"Saved FAISS index → {FAISS_FILE}")



text = read_pdf(PDF_PATH)

chunks = chunk_text(text)

build_embeddings(chunks)
