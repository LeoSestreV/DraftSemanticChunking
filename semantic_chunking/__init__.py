

from .preprocessing import (
    TextPreprocessor,
    Sentence,
    load_biography,
    preprocess_biography
)

from .embeddings import (
    OllamaEmbedder,
    EmbeddedSentence,
    embed_sentences
)

from .chunking import (
    SemanticChunker,    
    ChunkingStrategy,
    Chunk,
    chunk_sentences,
    print_chunks_summary
)

from .main import (
    SemanticChunkingPipeline,
    ChunkingResult
)

__version__ = "0.1.0"

__all__ = [
    "TextPreprocessor",
    "Sentence",
    "load_biography",
    "preprocess_biography",
    "OllamaEmbedder",
    "EmbeddedSentence",
    "embed_sentences",
    "SemanticChunker",
    "ChunkingStrategy",
    "Chunk",
    "chunk_sentences",
    "print_chunks_summary",
    "SemanticChunkingPipeline",
    "ChunkingResult",
]