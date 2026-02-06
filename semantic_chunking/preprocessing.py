import re
from dataclasses import dataclass
from typing import List, Optional
import spacy

@dataclass
class Sentence:
    """Représente une phrase segmentée avec métadonnées."""
    text: str
    index: int
    start_char: int
    end_char: int

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Sentence({self.index}: '{preview}')"

class TextPreprocessor:
    def __init__(self, spacy_model: str = "fr_core_news_lg"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            # Repli sur le petit modèle si le large n'est pas installé
            self.nlp = spacy.load("fr_core_news_sm")
        self.nlp.max_length = 2_000_000

    def clean_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'(\w+)-\s*\n\s*([a-zàâäéèêëïîôùûüœæç])', r'\1\2', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = text.replace('«', '"').replace('»', '"').replace('“', '"').replace('”', '"')
        text = text.replace('’', "'").replace('‘', "'")
        return self._fix_ocr_errors(text)

    def _fix_ocr_errors(self, text: str) -> str:
        text = re.sub(r'(\w)11(\w)', r'\1ll\2', text)
        text = re.sub(r'\s+([,;:\.!\?])', r'\1', text)
        text = re.sub(r'([,;])([A-ZÀ-Ü])', r'\1 \2', text)
        return text

    def segment_sentences(self, text: str, min_length: int = 10) -> List[Sentence]:
        doc = self.nlp(text)
        sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # On garde les phrases courtes SI elles commencent par une majuscule (titres)
            if len(sent_text) < min_length and not (sent_text and sent_text[0].isupper()):
                continue
            if not re.search(r'[a-zA-ZàâäéèêëïîôùûüœæçÀÂÄÉÈÊËÏÎÔÙÛÜŒÆÇ]', sent_text):
                continue
            sentences.append(Sentence(
                text=sent_text, index=len(sentences),
                start_char=sent.start_char, end_char=sent.end_char
            ))
        return sentences

    def process(self, text: str, min_sentence_length: int = 10) -> List[Sentence]:
        cleaned = self.clean_text(text)
        return self.segment_sentences(cleaned, min_length=min_sentence_length)

def load_biography(filepath: str, encoding: str = "utf-8") -> str:
    """Charge le contenu brut d'un fichier texte."""
    with open(filepath, 'r', encoding=encoding) as f:
        return f.read()

def preprocess_biography(filepath: str, spacy_model: str = "fr_core_news_sm") -> List[Sentence]:
    """Fonction utilitaire pour charger et prétraiter un fichier."""
    preprocessor = TextPreprocessor(spacy_model)
    text = load_biography(filepath)
    return preprocessor.process(text)