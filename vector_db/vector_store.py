import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.embeddings = None

    def build_index(self, texts):
        self.texts = texts
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings, dtype=np.float32))

    def load_index(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.texts = data['texts']
            self.embeddings = data['embeddings']
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(np.array(self.embeddings, dtype=np.float32))
        print(f"Loaded vector index with {len(self.texts)} texts.")

    def search(self, query, top_k=5):
        if self.index is None:
            raise ValueError("Vector index is not loaded. Call load_index() first.")
        query_emb = self.model.encode([query])
        D, I = self.index.search(np.array(query_emb, dtype=np.float32), top_k)
        return [(self.texts[i], float(D[0][idx])) for idx, i in enumerate(I[0])] 