"""
Document splitting and utility helpers for industrial text extraction.
"""

import re
from typing import List

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_text_to_sentences(text: str) -> List[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    sentences = SENTENCE_SPLIT_RE.split(cleaned)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []

    text = normalize_text(text)
    sentences = split_text_to_sentences(text)
    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        if not current_chunk:
            current_chunk = sentence
            continue

        candidate = f"{current_chunk} {sentence}".strip()
        if len(candidate) <= max_chars:
            current_chunk = candidate
            continue

        chunks.append(current_chunk)
        if len(sentence) > max_chars:
            start = 0
            while start < len(sentence):
                end = start + max_chars
                chunk_piece = sentence[start:end].strip()
                if chunk_piece:
                    chunks.append(chunk_piece)
                start = end - overlap
                if start < 0:
                    start = 0
            current_chunk = ""
        else:
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
