"""
Configuration settings for the Semantic Chunking Pipeline.

This module contains default configurations and recommendations
for processing Belgian academic biographies in French.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
   
    spacy_model: str = "fr_core_news_sm"

    min_sentence_length: int = 10

    default_encoding: str = "utf-8"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    ollama_base_url: str = "http://localhost:11434"
    request_timeout: int = 30

   
    default_model: str = "nomic-embed-text"


@dataclass
class ChunkingConfig:
    """
    Configuration for semantic chunking.

    Threshold Guidelines for French Biographical Texts:
    ---------------------------------------------------
    - 0.3-0.4: Very large chunks, loose semantic grouping
               Use when you want broad thematic sections
    - 0.5:     Balanced (default), good for most biographies
               Creates chunks around life phases/topics
    - 0.6-0.7: Tighter semantic grouping, smaller chunks
               Use for detailed analysis or shorter texts
    - 0.8+:    Very small chunks, almost sentence-level
               Rarely useful for RAG, may lose context

    Chunk Size Guidelines:
    ----------------------
    For RAG systems, optimal chunk sizes are typically:
    - 512-1024 tokens (roughly 300-600 words)
    - 3-8 sentences for biographical content

    The min/max sentence settings help enforce reasonable sizes.
    """
    
    default_strategy: str = "sliding_window"

    
    similarity_threshold: float = 0.55

    min_chunk_size: int = 2  
    max_chunk_size: int = 8

    
    window_size: int = 2


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "preprocessing": {
                "spacy_model": self.preprocessing.spacy_model,
                "min_sentence_length": self.preprocessing.min_sentence_length,
                "default_encoding": self.preprocessing.default_encoding
            },
            "embedding": {
                "ollama_base_url": self.embedding.ollama_base_url,
                "request_timeout": self.embedding.request_timeout,
                "default_model": self.embedding.default_model
            },
            "chunking": {
                "default_strategy": self.chunking.default_strategy,
                "similarity_threshold": self.chunking.similarity_threshold,
                "min_chunk_size": self.chunking.min_chunk_size,
                "max_chunk_size": self.chunking.max_chunk_size,
                "window_size": self.chunking.window_size
            }
        }



DEFAULT_CONFIG = PipelineConfig()

SHORT_BIO_CONFIG = PipelineConfig(
    chunking=ChunkingConfig(
        similarity_threshold=0.6,  
        min_chunk_size=1,          
        max_chunk_size=5
    )
)

LONG_BIO_CONFIG = PipelineConfig(
    chunking=ChunkingConfig(
        similarity_threshold=0.45,  
        min_chunk_size=3,
        max_chunk_size=15
    )
)

HIGH_ACCURACY_CONFIG = PipelineConfig(
    preprocessing=PreprocessingConfig(
        spacy_model="fr_core_news_lg"  
    ),
    embedding=EmbeddingConfig(
        default_model="mxbai-embed-large"  
    ),
    chunking=ChunkingConfig(
        similarity_threshold=0.55,
        window_size=2  
    )
)


EMBEDDING_MODEL_RECOMMENDATIONS = """
Embedding Model Recommendations for French Biographical Texts
=============================================================

1. nomic-embed-text (RECOMMENDED)
   - Dimensions: 768
   - Speed: Fast
   - Quality: Good
   - Multilingual: Yes
   - Best for: General use, balanced performance
   - Install: ollama pull nomic-embed-text

2. mxbai-embed-large
   - Dimensions: 1024
   - Speed: Medium
   - Quality: Very Good
   - Multilingual: Yes
   - Best for: When accuracy is priority over speed
   - Install: ollama pull mxbai-embed-large

3. snowflake-arctic-embed
   - Dimensions: 1024
   - Speed: Fast
   - Quality: Good
   - Multilingual: Yes
   - Best for: Large corpora, batch processing
   - Install: ollama pull snowflake-arctic-embed

Note: All models work well with French text. The default
(nomic-embed-text) is recommended for most use cases as it
provides the best balance of quality and speed.
"""
