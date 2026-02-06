from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

class ChunkingStrategy(Enum):
    SLIDING_WINDOW = "sliding_window"
    HIERARCHICAL = "hierarchical"

@dataclass
class Chunk:
    sentences: List = field(default_factory=list)
    chunk_id: int = 0
    start_sentence_idx: int = 0
    end_sentence_idx: int = 0
    coherence_score: float = 0.0
    start_char: int = 0
    end_char: int = 0

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)

    @property
    def char_length(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "num_sentences": self.num_sentences,
            "char_length": self.char_length,
            "start_sentence_idx": self.start_sentence_idx,
            "end_sentence_idx": self.end_sentence_idx,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "coherence_score": round(self.coherence_score, 4)
        }

class SemanticChunker:
    def __init__(self, similarity_threshold: float = 0.5, min_chunk_size: int = 2, max_chunk_size: int = 10):
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _compute_chunk_coherence(self, embeddings: List[np.ndarray]) -> float:
        if len(embeddings) < 2: return 1.0
        mat = np.vstack(embeddings)
        sims = sklearn_cosine(mat)
        return float(np.mean(sims[np.triu_indices(len(embeddings), k=1)]))

    def chunk(self, embedded_sentences: List, strategy: ChunkingStrategy = ChunkingStrategy.SLIDING_WINDOW, **kwargs):
        return self.chunk_sliding_window(embedded_sentences, window_size=2)

    def chunk_sliding_window(self, embedded_sentences: List, window_size: int = 2) -> List[Chunk]:
        if not embedded_sentences: return []
        chunks = []
        curr_sents = [embedded_sentences[0]]
        curr_embs = [embedded_sentences[0].embedding]

        for i in range(1, len(embedded_sentences)):
            sent = embedded_sentences[i]
            window = curr_embs[-window_size:]
            sims = [float(sklearn_cosine([sent.embedding], [e])[0,0]) for e in window]
            avg_sim = np.mean(sims)

            if (avg_sim < self.similarity_threshold or len(curr_sents) >= self.max_chunk_size) and len(curr_sents) >= self.min_chunk_size:
                chunks.append(Chunk(
                    sentences=curr_sents.copy(), chunk_id=len(chunks),
                    start_sentence_idx=curr_sents[0].index, end_sentence_idx=curr_sents[-1].index,
                    coherence_score=self._compute_chunk_coherence(curr_embs),
                    start_char=curr_sents[0].start_char, end_char=curr_sents[-1].end_char
                ))
                curr_sents, curr_embs = [sent], [sent.embedding]
            else:
                curr_sents.append(sent)
                curr_embs.append(sent.embedding)

        if curr_sents:
            chunks.append(Chunk(sentences=curr_sents, chunk_id=len(chunks), coherence_score=1.0))
        return chunks

def chunk_sentences(embedded_sentences: List, **kwargs) -> List[Chunk]:
    chunker = SemanticChunker(
        similarity_threshold=kwargs.get('similarity_threshold', 0.5),
        min_chunk_size=kwargs.get('min_chunk_size', 2),
        max_chunk_size=kwargs.get('max_chunk_size', 10)
    )
    return chunker.chunk(embedded_sentences)

def print_chunks_summary(chunks: List[Chunk]):
    print(f"Total chunks: {len(chunks)}")