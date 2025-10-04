from src.retriever import load_corpus, retrieve
from src.generator import build_prompt, generate_with_ollama


class RAGPipeline:
    def __init__(self, model="mistral:7b", save_dir="vectordb"):
        self.chunks, self.embs, self.index, self.embedder = load_corpus()
        self.model = model

    def ask(self, query, top_k=3):
        retrieved = retrieve(query, self.chunks, self.index, self.embedder, top_k=top_k)
        prompt = build_prompt(query, retrieved)
        return generate_with_ollama(prompt, model=self.model)
