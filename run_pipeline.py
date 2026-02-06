import json, os
from pathlib import Path
from dataclasses import asdict
from semantic_chunking.main import SemanticChunkingPipeline

def run():
    root = Path(__file__).parent
    in_dir, out_dir = root / "BioTxt", root / "output"
    out_dir.mkdir(exist_ok=True)
    
    pipeline = SemanticChunkingPipeline()
    
    files = list(in_dir.glob("*.txt"))
    
    for f in files:
        dest = out_dir / f"{f.stem}.json"
        
        if not dest.exists():
            try:
                result = pipeline.process_file(str(f))
                with open(dest, "w", encoding="utf-8") as out:
                    json.dump(asdict(result), out, ensure_ascii=False, indent=2)
            except Exception:
                pass 

if __name__ == "__main__":
    run()