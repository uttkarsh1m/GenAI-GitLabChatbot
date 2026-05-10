"""
Text Chunking Module
Splits scraped content into optimal chunks for embedding and retrieval.
"""

import re
import json
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks by sentence boundaries.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in words
        overlap: Number of words to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Sentence splitter that avoids splitting on abbreviations, decimals, URLs
    # Uses a negative lookbehind for common abbreviations
    sentence_pattern = re.compile(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\!|\?)\s+'
    )
    sentences = sentence_pattern.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]

    if not sentences:
        # Fallback: split by double newline or just return the whole text as one chunk
        fallback = text.strip()
        return [fallback] if len(fallback.split()) >= 10 else []

    chunks = []
    current_chunk: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # If a single sentence exceeds chunk_size, split it by words
        if word_count > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            words = sentence.split()
            for i in range(0, len(words), chunk_size - overlap):
                sub = ' '.join(words[i:i + chunk_size])
                if len(sub.split()) >= 10:
                    chunks.append(sub)
            continue

        if current_word_count + word_count > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))

            # Keep overlap sentences
            overlap_sentences: List[str] = []
            overlap_count = 0
            for s in reversed(current_chunk):
                wc = len(s.split())
                if overlap_count + wc <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_count += wc
                else:
                    break

            current_chunk = overlap_sentences
            current_word_count = overlap_count

        current_chunk.append(sentence)
        current_word_count += word_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return [c for c in chunks if len(c.split()) >= 10]


def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Remove duplicate and near-duplicate chunks.

    Two levels of deduplication:
    1. Exact: identical text (SHA-256 hash)
    2. Near-duplicate: same first 120 characters (catches overlap artifacts
       where two chunks start identically but differ slightly at the end)

    When a duplicate is found, the first occurrence is kept (preserves
    the chunk with the lower chunk_id and better source metadata).

    Returns:
        Deduplicated list with reassigned sequential chunk_ids.
    """
    import hashlib

    seen_hashes: set = set()
    seen_prefixes: set = set()
    unique_chunks = []
    exact_removed = 0
    near_removed = 0

    for chunk in chunks:
        text = chunk.get("text") or ""
        if not text:
            continue

        # Level 1: exact hash
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in seen_hashes:
            exact_removed += 1
            continue
        seen_hashes.add(text_hash)

        # Level 2: near-duplicate prefix (first 120 chars, normalised)
        prefix = " ".join(text[:120].lower().split())
        if prefix in seen_prefixes:
            near_removed += 1
            continue
        seen_prefixes.add(prefix)

        unique_chunks.append(chunk)

    # Reassign sequential chunk_ids so FAISS index aligns correctly
    for new_id, chunk in enumerate(unique_chunks):
        chunk["chunk_id"] = new_id

    total_removed = exact_removed + near_removed
    if total_removed:
        logger.info(
            f"Deduplication: removed {exact_removed} exact + {near_removed} near-duplicates "
            f"({total_removed} total). {len(unique_chunks)} unique chunks remain."
        )

    return unique_chunks


def create_chunks_from_documents(documents: List[Dict]) -> List[Dict]:
    """
    Convert scraped documents into searchable chunks with metadata.

    Each chunk contains:
    - text: The actual content
    - source_url: Where it came from
    - title: Page title
    - section: Section heading
    - category: handbook or direction
    - chunk_id: Unique identifier
    """
    if not documents:
        logger.warning("No documents provided to chunker.")
        return []

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        if not isinstance(doc, dict):
            continue

        url = doc.get("url") or ""
        title = doc.get("title") or "Unknown"
        category = doc.get("category") or "general"
        sections = doc.get("sections") or []
        full_text = doc.get("full_text") or ""

        if sections:
            for section in sections:
                if not isinstance(section, dict):
                    continue
                heading = section.get("heading") or ""
                content = (section.get("content") or "").strip()

                if not content or len(content) < 50:
                    continue

                full_section_text = f"{heading}: {content}" if heading else content
                text_chunks = chunk_text(full_section_text, chunk_size=400, overlap=40)

                for chunk_text_item in text_chunks:
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text_item,
                        "source_url": url,
                        "title": title,
                        "section": heading,
                        "category": category,
                        "word_count": len(chunk_text_item.split())
                    })
                    chunk_id += 1

        elif full_text and len(full_text) > 100:
            text_chunks = chunk_text(full_text, chunk_size=400, overlap=40)
            for chunk_text_item in text_chunks:
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text_item,
                    "source_url": url,
                    "title": title,
                    "section": "",
                    "category": category,
                    "word_count": len(chunk_text_item.split())
                })
                chunk_id += 1

    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents.")
    all_chunks = deduplicate_chunks(all_chunks)
    return all_chunks


def save_chunks(chunks: List[Dict], path: str = "data/chunks.json") -> bool:
    """Save processed chunks to disk. Returns True on success."""
    if not chunks:
        logger.warning("No chunks to save.")
        return False
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to {path}")
        return True
    except OSError as e:
        logger.error(f"Failed to save chunks: {e}")
        return False


def load_chunks(path: str = "data/chunks.json") -> List[Dict]:
    """Load chunks from disk. Returns empty list on any failure."""
    if not os.path.exists(path):
        logger.warning(f"Chunks file not found: {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("Chunks file has unexpected format.")
            return []
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load chunks: {e}")
        return []


def get_chunk_stats(chunks: List[Dict]) -> Dict:
    """Return statistics about the chunks."""
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_words": 0,
            "min_words": 0,
            "max_words": 0,
            "by_category": {},
            "unique_sources": 0
        }

    word_counts = [c.get("word_count", 0) for c in chunks]
    categories: Dict[str, int] = {}
    for c in chunks:
        cat = c.get("category") or "unknown"
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_chunks": len(chunks),
        "avg_words": round(sum(word_counts) / len(word_counts), 1),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "by_category": categories,
        "unique_sources": len(set(c.get("source_url", "") for c in chunks if c.get("source_url")))
    }
