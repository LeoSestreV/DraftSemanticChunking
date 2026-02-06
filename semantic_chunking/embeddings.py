import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor

@dataclass
class EmbeddedSentence:
    text: str
    index: int
    embedding: np.ndarray
    start_char: int
    end_char: int

    def __repr__(self) -> str:
        return f"EmbeddedSentence({self.index}, dim={len(self.embedding)})"

class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434", timeout: int = 30):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def _make_request(self, endpoint: str, data: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        payload = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            raise RuntimeError(f"Ollama non joignable: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        response = self._make_request("/api/embeddings", {"model": self.model, "prompt": text})
        vec = response.get("embedding") or response.get("embeddings", [None])[0]
        return np.array(vec, dtype=np.float32)

    def get_embeddings_batch(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        if show_progress: print(f"  Génération des embeddings (Parallel)...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            embeddings = list(executor.map(self.get_embedding, texts))
        return embeddings

def embed_sentences(sentences: List, model: str = "nomic-embed-text", show_progress: bool = True) -> List[np.ndarray]:
    embedder = OllamaEmbedder(model=model)
    texts = [s.text if hasattr(s, 'text') else s for s in sentences]
    return embedder.get_embeddings_batch(texts, show_progress=show_progress)