"""Thin client for the local Ollama REST API.

Uses one pooled requests.Session so every call reuses an open TCP connection to Ollama
(no per-request handshake).  Only `requests` is needed, which ships with Raspberry Pi OS;
the official `ollama` Python package sends exactly the same JSON to the same endpoints.
"""
import json
import time

import requests
from requests.adapters import HTTPAdapter


class Ollama:
    def __init__(self, base_url="http://127.0.0.1:11434", pool=8, timeout=600):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool)
        self.s.mount("http://", adapter)

    # ---- info -------------------------------------------------------------------------
    def version(self):
        return self.s.get(f"{self.base}/api/version", timeout=5).json()["version"]

    def loaded(self):
        """Models currently resident in RAM (GET /api/ps)."""
        return [m["name"] for m in self.s.get(f"{self.base}/api/ps", timeout=5).json().get("models", [])]

    # ---- chat -------------------------------------------------------------------------
    def chat_stream(self, model, messages, options=None, keep_alive=-1):
        """Yield Ollama's streamed chunks (dicts).  The last chunk has done=True and the timing stats.

        Closing the generator early (client went away) closes the HTTP response, and Ollama
        cancels the generation, so an abandoned request stops using the CPU straight away."""
        body = {"model": model, "messages": messages, "stream": True, "keep_alive": keep_alive}
        if options:
            body["options"] = options
        r = self.s.post(f"{self.base}/api/chat", json=body, stream=True, timeout=self.timeout)
        try:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise RuntimeError(chunk["error"])
                    yield chunk
        finally:
            r.close()

    def chat(self, model, messages, options=None, keep_alive=-1):
        body = {"model": model, "messages": messages, "stream": False, "keep_alive": keep_alive}
        if options:
            body["options"] = options
        r = self.s.post(f"{self.base}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---- embeddings -------------------------------------------------------------------
    def embed(self, model, texts, keep_alive=-1):
        r = self.s.post(f"{self.base}/api/embed",
                        json={"model": model, "input": texts, "keep_alive": keep_alive},
                        timeout=self.timeout)
        r.raise_for_status()
        return r.json()["embeddings"]

    # ---- warm-up ----------------------------------------------------------------------
    def warmup(self, chat_model, embed_model=None):
        """Load the models into RAM once at start-up (keep_alive=-1 keeps them there), so the
        first user request does not pay the ~3 s model-load time."""
        t = time.perf_counter()
        self.s.post(f"{self.base}/api/generate", json={"model": chat_model, "keep_alive": -1},
                    timeout=self.timeout).raise_for_status()
        if embed_model:
            self.embed(embed_model, ["warm-up"])
        return time.perf_counter() - t
