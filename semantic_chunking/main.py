import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import sys

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from .preprocessing import TextPreprocessor, load_biography, Sentence
from .embeddings import OllamaEmbedder, embed_sentences
from .chunking import SemanticChunker, ChunkingStrategy, chunk_sentences

@dataclass
class ChunkingResult:
    source_file: str
    processed_at: str
    total_sentences: int
    total_chunks: int
    chunks: List[Dict[str, Any]]
    config: Dict[str, Any]
    avg_coherence: float
    avg_chunk_size: float

class SemanticChunkingPipeline:
    def __init__(self, spacy_model: str = "fr_core_news_sm", min_sentence_length: int = 10,
                 ollama_model: str = "nomic-embed-text", chunking_strategy: str = "sliding_window",
                 similarity_threshold: float = 0.5, min_chunk_size: int = 2, max_chunk_size: int = 10):
        self.config = {
            "spacy_model": spacy_model,
            "min_sentence_length": min_sentence_length,
            "ollama_model": ollama_model,
            "chunking_strategy": chunking_strategy,
            "similarity_threshold": similarity_threshold,
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size
        }
        self.preprocessor = TextPreprocessor(spacy_model)
        self.chunker = SemanticChunker(similarity_threshold=similarity_threshold, 
                                      min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size)
        self.strategy = ChunkingStrategy(chunking_strategy)

    def process_text(self, text: str, source_name: str = "unknown", show_progress: bool = True) -> ChunkingResult:
        sentences = self.preprocessor.process(text, min_sentence_length=self.config["min_sentence_length"])
        embedded = embed_sentences(sentences, model=self.config["ollama_model"], show_progress=show_progress)
        
        from .embeddings import EmbeddedSentence
        embedded_objects = []
        for i, (sent, emb) in enumerate(zip(sentences, embedded)):
            embedded_objects.append(EmbeddedSentence(
                text=sent.text, index=sent.index, embedding=emb,
                start_char=sent.start_char, end_char=sent.end_char
            ))

        chunks = self.chunker.chunk(embedded_objects, strategy=self.strategy)
        
        avg_coherence = sum(c.coherence_score for c in chunks) / len(chunks) if chunks else 0
        avg_size = sum(c.num_sentences for c in chunks) / len(chunks) if chunks else 0

        return ChunkingResult(
            source_file=source_name, processed_at=datetime.now().isoformat(),
            total_sentences=len(sentences), total_chunks=len(chunks),
            chunks=[c.to_dict() for c in chunks], config=self.config,
            avg_coherence=round(avg_coherence, 4), avg_chunk_size=round(avg_size, 2)
        )

    def process_file(self, filepath: str, encoding: str = "utf-8", show_progress: bool = True) -> ChunkingResult:
        text = load_biography(filepath, encoding)
        return self.process_text(text, source_name=os.path.basename(filepath), show_progress=show_progress)

    def process_directory(self, dirpath: str, output_dir: str = "output"):
        os.makedirs(output_dir, exist_ok=True)
        files = list(Path(dirpath).glob("*.txt"))
        print(f"Traitements de {len(files)} fichiers...")
        
        for f in files:
            print(f"En cours : {f.name}")
            result = self.process_file(str(f))
            output_file = os.path.join(output_dir, f.stem + ".json")
            with open(output_file, "w", encoding="utf-8") as out:
                json.dump(asdict(result), out, ensure_ascii=False, indent=2)
            print(f"Sauvegardé dans : {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Fichier ou dossier")
    parser.add_argument("--batch", action="store_true", help="Mode dossier")
    parser.add_argument("--model", default="nomic-embed-text")
    args = parser.parse_args()

    pipeline = SemanticChunkingPipeline(ollama_model=args.model)

    if args.batch:
        pipeline.process_directory(args.input)
    else:
        result = pipeline.process_file(args.input)
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))

def export_results(results: List[ChunkingResult], output_path: str, format: str = "json") -> None:
    """
    Ancienne fonction d'export, conservée pour la compatibilité des imports.
    Le nouvel export est maintenant intégré à process_directory.
    """
    pass

if __name__ == "__main__":
    main()