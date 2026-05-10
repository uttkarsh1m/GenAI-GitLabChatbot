"""
Embeddings & Vector Store Module
Creates and manages FAISS vector index for semantic search.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid slow startup
_embedding_model = None
_reranker_model = None
_faiss = None


def get_embedding_model():
    """Lazy-load the sentence transformer model (lighter L3 variant)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model...")
        _embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        logger.info("Embedding model loaded.")
    return _embedding_model


def get_reranker():
    """
    Lazy-load the cross-encoder reranker (TinyBERT variant — 5x faster load).
    Falls back gracefully if unavailable.
    """
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model...")
            _reranker_model = CrossEncoder(
                'cross-encoder/ms-marco-TinyBERT-L-2-v2',
                max_length=512
            )
            logger.info("Reranker model loaded.")
        except Exception as e:
            logger.warning(f"Reranker unavailable ({e}). Falling back to FAISS order.")
            _reranker_model = "unavailable"
    return _reranker_model if _reranker_model != "unavailable" else None


def get_faiss():
    """Lazy-load FAISS."""
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Generate embeddings — uses model server if running, else direct model."""
    if not texts:
        logger.warning("No texts provided for embedding.")
        return np.array([]).reshape(0, 384)

    try:
        from src.model_server import ModelClient
        client = ModelClient()
        if client._is_server_up():
            return client.encode(texts)
    except Exception:
        pass

    # Fallback: direct model
    model = get_embedding_model()
    logger.info(f"Embedding {len(texts)} texts...")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(embeddings)
        except Exception as e:
            logger.error(f"Failed to embed batch {i}-{i+batch_size}: {e}")

    if not all_embeddings:
        logger.error("All embedding batches failed!")
        return np.array([]).reshape(0, 384)

    return np.vstack(all_embeddings).astype('float32')


def build_index(chunks: List[Dict], index_path: str = "data/faiss_index",
                chunks_path: str = "data/chunks.json") -> Tuple:
    """
    Build a FAISS index from document chunks.
    
    Args:
        chunks: List of chunk dicts with 'text' field
        index_path: Where to save the FAISS index
        chunks_path: Where to save chunk metadata
    
    Returns:
        Tuple of (faiss_index, chunks_list) or (None, None) on failure
    """
    if not chunks:
        logger.error("Cannot build index: no chunks provided.")
        return None, None

    try:
        faiss = get_faiss()
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)

        # Extract texts safely
        texts = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get('text')
            if text and isinstance(text, str) and text.strip():
                texts.append(text)
            else:
                logger.warning(f"Skipping chunk {chunk.get('chunk_id', '?')}: missing or empty text")

        if not texts:
            logger.error("No valid texts found in chunks!")
            return None, None

        embeddings = embed_texts(texts)
        if embeddings.size == 0:
            logger.error("Embedding failed: no embeddings generated.")
            return None, None

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save index
        faiss.write_index(index, f"{index_path}.bin")

        # Save chunks metadata
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Built FAISS index with {index.ntotal} vectors. Saved to {index_path}.bin")
        return index, chunks

    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        return None, None


def load_index(index_path: str = "data/faiss_index",
               chunks_path: str = "data/chunks.json") -> Tuple:
    """
    Load a previously built FAISS index.
    
    Returns:
        Tuple of (faiss_index, chunks_list) or (None, None) if not found/corrupted
    """
    try:
        faiss = get_faiss()
        bin_path = f"{index_path}.bin"

        if not os.path.exists(bin_path):
            logger.warning(f"Index file not found: {bin_path}")
            return None, None
        if not os.path.exists(chunks_path):
            logger.warning(f"Chunks file not found: {chunks_path}")
            return None, None

        # Load index
        index = faiss.read_index(bin_path)
        if index.ntotal == 0:
            logger.warning("Index is empty!")
            return None, None

        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        if not isinstance(chunks, list) or not chunks:
            logger.error("Chunks file is empty or invalid format.")
            return None, None

        if index.ntotal != len(chunks):
            logger.warning(f"Index/chunks mismatch: {index.ntotal} vectors vs {len(chunks)} chunks")

        logger.info(f"Loaded FAISS index with {index.ntotal} vectors and {len(chunks)} chunks.")
        return index, chunks

    except (OSError, json.JSONDecodeError, RuntimeError) as e:
        logger.error(f"Failed to load index: {e}")
        return None, None


def rerank(query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    """
    Rerank FAISS candidates using a cross-encoder model.
    Uses model server if running (near-instant), falls back to direct model.
    """
    if not candidates:
        return candidates

    # Build (query, passage) pairs once
    pairs = [(query.strip()[:512], c.get('text', '')[:512]) for c in candidates]

    try:
        # Try model server first — already loaded, sub-millisecond
        from src.model_server import ModelClient
        client = ModelClient()
        if client._is_server_up():
            scores = client.rerank(pairs)
            for chunk, score in zip(candidates, scores):
                chunk['rerank_score'] = float(score)
                faiss_conf = chunk.get('confidence', 0) / 100.0
                blended = 0.7 * _sigmoid(float(score)) + 0.3 * faiss_conf
                chunk['confidence'] = round(blended * 100, 2)
            reranked = sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
            return reranked[:top_k]
    except Exception:
        pass

    # Fallback: direct reranker model
    reranker = get_reranker()
    if reranker is None:
        return candidates[:top_k]

    try:
        scores = reranker.predict(pairs, show_progress_bar=False)
        for chunk, score in zip(candidates, scores):
            chunk['rerank_score'] = float(score)
            faiss_conf = chunk.get('confidence', 0) / 100.0
            blended = 0.7 * _sigmoid(float(score)) + 0.3 * faiss_conf
            chunk['confidence'] = round(blended * 100, 2)
        reranked = sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
        return reranked[:top_k]

    except Exception as e:
        logger.warning(f"Reranking failed ({e}), using FAISS order.")
        return candidates[:top_k]


def _sigmoid(x: float) -> float:
    """Sigmoid to normalise cross-encoder logit scores to [0, 1]."""
    import math
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def search(query: str, index, chunks: List[Dict], top_k: int = 5,
           category_filter: Optional[str] = None,
           use_reranker: bool = True) -> List[Dict]:
    """
    Two-stage retrieval: FAISS bi-encoder → cross-encoder reranker.

    Stage 1 (FAISS): Fast approximate nearest-neighbour search.
      Retrieves top_k * 3 candidates using vector similarity.
    Stage 2 (Reranker): Cross-encoder scores each (query, chunk) pair
      jointly and reorders by true relevance. Falls back to FAISS
      order if the reranker is unavailable.

    Args:
        query: User's question
        index: FAISS index
        chunks: List of chunk metadata
        top_k: Final number of results to return
        category_filter: Optional filter ('handbook' or 'direction')
        use_reranker: Set False to skip reranking (e.g. Search Explorer)

    Returns:
        List of relevant chunks, reranked, with similarity + rerank scores
    """
    # Guard: validate inputs
    if not query or not query.strip():
        logger.warning("Empty query passed to search.")
        return []
    if index is None:
        logger.error("Search called with None index.")
        return []
    if not chunks:
        logger.error("Search called with empty chunks list.")
        return []

    try:
        query_trimmed = query.strip()[:512]

        # Use model server if available (already loaded — fast)
        query_embedding = embed_texts([query_trimmed])
        if query_embedding.size == 0:
            return []

        # Search with extra results for filtering
        search_k = min(top_k * 3 if category_filter else top_k, index.ntotal)
        if search_k == 0:
            return []

        scores, indices = index.search(query_embedding, search_k)

        # Stage 1 complete — collect all FAISS candidates before filtering
        faiss_candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            score_val = float(score)
            if not (score_val == score_val):  # NaN check
                score_val = 0.0
            score_val = max(0.0, min(score_val, 1.0))

            chunk = chunks[idx].copy()
            chunk['similarity_score'] = score_val
            chunk['confidence'] = round(score_val * 100, 2)

            if category_filter and chunk.get('category') != category_filter:
                continue

            faiss_candidates.append(chunk)

        # Stage 2: rerank the candidates
        if use_reranker and faiss_candidates:
            return rerank(query, faiss_candidates, top_k)

        return faiss_candidates[:top_k]

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def index_exists(index_path: str = "data/faiss_index") -> bool:
    """Check if a built index exists."""
    return os.path.exists(f"{index_path}.bin")


# ─── BM25 Index ───────────────────────────────────────────────────────────────

_bm25_index = None
_bm25_chunks: List[Dict] = []


def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    Lowercases, removes punctuation, splits on whitespace.
    Keeps meaningful tokens only (length > 1).
    """
    import re
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if len(t) > 1]


def build_bm25_index(chunks: List[Dict]) -> object:
    """Build and cache a BM25 index from chunks."""
    global _bm25_index, _bm25_chunks
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(c.get('text', '')) for c in chunks]
    _bm25_index = BM25Okapi(corpus)
    _bm25_chunks = chunks
    logger.info(f"Built BM25 index over {len(chunks)} chunks.")
    return _bm25_index


def get_bm25_index(chunks: List[Dict]) -> object:
    """Return cached BM25 index, building it if needed."""
    # Rebuild if chunks count changed (new ingestion) or not yet built
    if _bm25_index is None or len(_bm25_chunks) != len(chunks):
        build_bm25_index(chunks)
    return _bm25_index


def bm25_search(query: str, chunks: List[Dict], top_k: int = 15) -> List[Dict]:
    """
    BM25 keyword search over chunks.

    Returns top_k chunks with a normalised bm25_score in [0, 1].
    """
    if not query or not chunks:
        return []

    try:
        bm25 = get_bm25_index(chunks)
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = bm25.get_scores(tokens)

        # Normalise to [0, 1]
        max_score = float(max(scores)) if max(scores) > 0 else 1.0
        norm_scores = [float(s) / max_score for s in scores]

        # Collect top_k by score
        indexed = sorted(enumerate(norm_scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in indexed:
            if score <= 0:
                break
            chunk = chunks[idx].copy()
            chunk['bm25_score'] = round(score, 4)
            results.append(chunk)

        return results

    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
        return []


def hybrid_search(query: str, faiss_index, chunks: List[Dict],
                  top_k: int = 5,
                  category_filter: Optional[str] = None,
                  faiss_weight: float = 0.6,
                  bm25_weight: float = 0.4) -> List[Dict]:
    """
    Hybrid retrieval: BM25 keyword search + FAISS semantic search,
    fused with Reciprocal Rank Fusion (RRF) then reranked.

    Why RRF instead of score averaging?
    - BM25 and FAISS scores are on different scales
    - RRF uses rank positions (1/(k + rank)) which are scale-invariant
    - Proven to outperform score averaging in IR benchmarks

    Args:
        query: User question
        faiss_index: Built FAISS index
        chunks: Chunk metadata list
        top_k: Final results to return
        category_filter: Optional 'handbook' or 'direction'
        faiss_weight: Weight for FAISS ranks in fusion (default 0.6)
        bm25_weight: Weight for BM25 ranks in fusion (default 0.4)

    Returns:
        Reranked hybrid results with hybrid_score and confidence fields
    """
    if not query or faiss_index is None or not chunks:
        return []

    RRF_K = 60  # Standard RRF constant

    # ── Stage 1a: FAISS semantic search (get more candidates for fusion) ──
    faiss_results = search(query, faiss_index, chunks,
                           top_k=top_k * 4,
                           category_filter=category_filter,
                           use_reranker=False)  # rerank after fusion

    # ── Stage 1b: BM25 keyword search ─────────────────────────────────────
    bm25_results = bm25_search(query, chunks, top_k=top_k * 4)

    # Apply category filter to BM25 results
    if category_filter:
        bm25_results = [r for r in bm25_results
                        if r.get('category') == category_filter]

    # ── Stage 2: Reciprocal Rank Fusion ───────────────────────────────────
    rrf_scores: dict = {}

    for rank, chunk in enumerate(faiss_results):
        cid = chunk.get('chunk_id')
        if cid is None:
            continue
        rrf_scores[cid] = rrf_scores.get(cid, 0.0)
        rrf_scores[cid] += faiss_weight * (1.0 / (RRF_K + rank + 1))

    for rank, chunk in enumerate(bm25_results):
        cid = chunk.get('chunk_id')
        if cid is None:
            continue
        rrf_scores[cid] = rrf_scores.get(cid, 0.0)
        rrf_scores[cid] += bm25_weight * (1.0 / (RRF_K + rank + 1))

    # Build merged candidate list — preserve BM25 scores from BM25 results
    # Index BM25 results by chunk_id for fast lookup
    bm25_score_map = {c.get('chunk_id'): c.get('bm25_score', 0.0) for c in bm25_results}

    seen_ids: set = set()
    candidates: List[Dict] = []
    all_results = faiss_results + bm25_results

    for chunk in all_results:
        cid = chunk.get('chunk_id')
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        chunk = chunk.copy()
        chunk['hybrid_score'] = round(rrf_scores.get(cid, 0.0), 6)
        chunk['bm25_score'] = round(bm25_score_map.get(cid, 0.0), 4)
        # Blend hybrid score into confidence
        chunk['confidence'] = round(min(chunk['hybrid_score'] * 5000, 100), 2)
        candidates.append(chunk)

    # Sort by hybrid score
    candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
    top_candidates = candidates[:top_k * 2]

    # ── Stage 3: Cross-encoder reranking on fused candidates ──────────────
    return rerank(query, top_candidates, top_k)
