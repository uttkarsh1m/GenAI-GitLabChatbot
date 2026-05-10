"""
Model Server — loads embedding + reranker models once and keeps them in memory.
Communicates via a simple Unix socket so Streamlit can query without reloading.

Start: python src/model_server.py &
Query: ModelClient().encode(texts) / ModelClient().rerank(query, passages)
"""

import os
import sys
import json
import socket
import struct
import logging
import threading
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

SOCKET_PATH = "/tmp/gitbot_model_server.sock"
EMBED_MODEL  = "paraphrase-MiniLM-L3-v2"
RERANK_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"


# ─── Server ───────────────────────────────────────────────────────────────────

def _send_msg(conn, data: bytes):
    conn.sendall(struct.pack(">I", len(data)) + data)


def _recv_msg(conn) -> bytes:
    raw = conn.recv(4)
    if not raw:
        return b""
    length = struct.unpack(">I", raw)[0]
    chunks = []
    received = 0
    while received < length:
        chunk = conn.recv(min(65536, length - received))
        if not chunk:
            break
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def run_server():
    """Load models and serve requests over a Unix socket."""
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [ModelServer] %(message)s")
    logger.info("Loading models...")

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_embed  = ex.submit(SentenceTransformer, EMBED_MODEL)
        f_rerank = ex.submit(CrossEncoder, RERANK_MODEL, max_length=512)
        embed_model  = f_embed.result()
        rerank_model = f_rerank.result()

    logger.info("Models loaded. Listening on socket...")

    # Remove stale socket
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(10)
    os.chmod(SOCKET_PATH, 0o600)

    def handle(conn):
        try:
            raw = _recv_msg(conn)
            if not raw:
                return
            req = json.loads(raw.decode("utf-8"))
            op = req.get("op")

            if op == "encode":
                texts = req["texts"]
                embs = embed_model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )
                resp = {"embeddings": embs.tolist()}

            elif op == "rerank":
                pairs = req["pairs"]
                scores = rerank_model.predict(pairs, show_progress_bar=False)
                resp = {"scores": scores.tolist() if hasattr(scores, "tolist") else list(scores)}

            elif op == "ping":
                resp = {"status": "ok"}

            else:
                resp = {"error": f"Unknown op: {op}"}

            _send_msg(conn, json.dumps(resp).encode("utf-8"))
        except Exception as e:
            try:
                _send_msg(conn, json.dumps({"error": str(e)}).encode("utf-8"))
            except Exception:
                pass
        finally:
            conn.close()

    logger.info(f"Ready. Socket: {SOCKET_PATH}")
    while True:
        conn, _ = server.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


# ─── Client ───────────────────────────────────────────────────────────────────

class ModelClient:
    """
    Thin client that talks to the model server.
    Falls back to direct model loading if server is unavailable.
    """

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._fallback_embed  = None
        self._fallback_rerank = None

    def _is_server_up(self) -> bool:
        try:
            resp = self._call({"op": "ping"}, timeout=1.0)
            return resp.get("status") == "ok"
        except Exception:
            return False

    def _call(self, req: dict, timeout: float = None) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout or self.timeout)
        sock.connect(SOCKET_PATH)
        _send_msg(sock, json.dumps(req).encode("utf-8"))
        raw = _recv_msg(sock)
        sock.close()
        return json.loads(raw.decode("utf-8"))

    def encode(self, texts: list, normalize: bool = True) -> np.ndarray:
        if self._is_server_up():
            resp = self._call({"op": "encode", "texts": texts})
            if "embeddings" in resp:
                return np.array(resp["embeddings"], dtype="float32")
        # Fallback
        if self._fallback_embed is None:
            from sentence_transformers import SentenceTransformer
            self._fallback_embed = SentenceTransformer(EMBED_MODEL)
        return self._fallback_embed.encode(
            texts, normalize_embeddings=normalize, show_progress_bar=False
        ).astype("float32")

    def rerank(self, pairs: list) -> list:
        if self._is_server_up():
            resp = self._call({"op": "rerank", "pairs": pairs})
            if "scores" in resp:
                return resp["scores"]
        # Fallback
        if self._fallback_rerank is None:
            from sentence_transformers import CrossEncoder
            self._fallback_rerank = CrossEncoder(RERANK_MODEL, max_length=512)
        scores = self._fallback_rerank.predict(pairs, show_progress_bar=False)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)


def ensure_server_running():
    """Start the model server as a background process if not already running."""
    client = ModelClient()
    if client._is_server_up():
        return  # Already running

    import subprocess
    logger.info("Starting model server in background...")
    subprocess.Popen(
        [sys.executable, __file__],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )

    # Wait up to 60s for it to be ready
    import time
    for _ in range(60):
        time.sleep(1)
        if client._is_server_up():
            logger.info("Model server is ready.")
            return
    logger.warning("Model server did not start in time — using fallback.")


if __name__ == "__main__":
    run_server()
