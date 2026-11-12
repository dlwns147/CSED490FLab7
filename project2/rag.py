"""Retrieval-augmented generation (Project 2): a small knowledge base that runs on the Pi.

Pipeline
  ingest   : text files / web pages (Wikipedia via its API) -> ~100-word chunks with 20-word overlap  -> each chunk gets a header naming its document and location
  index    : dense vectors from `nomic-embed-text` (768-d, via Ollama) + a BM25 keyword index
  retrieve : dense top-k, BM25 top-k, or both fused with reciprocal-rank fusion (RRF, default)
  generate : the top chunks go into the prompt as numbered context; the model answers and cites [n]

Only numpy and the standard library are needed (both are on the Pi already).
"""
import hashlib
import html
import html.parser
import json
import math
import pathlib
import re
import threading
import time
from collections import Counter

import numpy as np
import requests

STOP = set("""a an and are as at be by can do does for from has have how i in is it its of on or
that the this to was what when where which who why will with you your""".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOP]


def windows(words, size=180, overlap=40):
    if len(words) <= size:
        return [words]
    step = size - overlap
    return [words[i:i + size] for i in range(0, max(len(words) - overlap, 1), step)]


class _HTMLText(html.parser.HTMLParser):
    """Visible text of an HTML page (drops script/style/nav)."""
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg"}

    def __init__(self):
        super().__init__()
        self.out, self.depth = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        elif tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth:
            self.out.append(data)


class BM25:
    def __init__(self, docs_tokens, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in docs_tokens]
        self.len = np.array([len(t) for t in docs_tokens], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(self.len) else 1.0
        df = Counter(w for t in docs_tokens for w in set(t))
        n = len(docs_tokens)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query):
        q = tokenize(query)
        s = np.zeros(len(self.tf), dtype=np.float32)
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for w in q:
                f = tf.get(w)
                if f:
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return s


class KnowledgeBase:
    DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "   # nomic-embed-text task prefixes

    def __init__(self, ollama, embed_model="nomic-embed-text", index_dir="index"):
        self.ol, self.embed_model = ollama, embed_model
        self.dir = pathlib.Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.chunks, self.emb, self.bm25 = [], np.zeros((0, 768), np.float32), None
        self._load()

    # ---------------------------------------------------------------- persistence
    def _load(self):
        cj, en = self.dir / "chunks.json", self.dir / "emb.npy"
        if cj.exists() and en.exists():
            self.chunks = json.loads(cj.read_text())
            self.emb = np.load(en)
            self._rebuild_bm25()

    def _save(self):
        (self.dir / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False))
        np.save(self.dir / "emb.npy", self.emb)

    def _rebuild_bm25(self):
        self.bm25 = BM25([tokenize(c["header"] + " " + c["text"]) for c in self.chunks])

    # ---------------------------------------------------------------- ingest
    def _embed(self, texts, prefix, batch=16):
        out = []
        for i in range(0, len(texts), batch):
            out += self.ol.embed(self.embed_model, [prefix + t for t in texts[i:i + batch]])
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)

    def _add(self, doc, pieces):
        """pieces: list of (location, text).  Replaces any earlier version of `doc`."""
        new = []
        for loc, text in pieces:
            text = text.strip()
            if len(text.split()) < 3:
                continue
            header = f"{doc} | {loc}" if loc else doc
            cid = hashlib.sha1(f"{doc}|{loc}|{text}".encode()).hexdigest()[:12]
            new.append({"id": cid, "doc": doc, "loc": loc, "header": header, "text": text})
        if not new:
            return 0
        vecs = self._embed([f"{c['header']}\n{c['text']}" for c in new], self.DOC_PREFIX)
        with self.lock:
            keep = [i for i, c in enumerate(self.chunks) if c["doc"] != doc]
            self.chunks = [self.chunks[i] for i in keep] + new
            self.emb = np.vstack([self.emb[keep], vecs]) if len(keep) else vecs
            self._rebuild_bm25()
            self._save()
        return len(new)

    def add_markdown(self, path):
        """A Markdown file whose '## ' sections become chunks (one section = one chunk)."""
        text = pathlib.Path(path).read_text()
        m = re.search(r"^# (.+)$", text, re.M)
        doc = m.group(1).strip() if m else pathlib.Path(path).stem
        pieces = []
        for sec in re.split(r"^## ", text, flags=re.M)[1:]:
            loc, _, body = sec.partition("\n")
            words = body.split()
            for j, w in enumerate(windows(words, 220, 40)):
                pieces.append((loc.strip() + (f" (part {j + 1})" if j else ""), " ".join(w)))
        return self._add(doc, pieces)

    def add_text(self, title, text, size=100, overlap=20):
        """~100-word windows (about one paragraph), so k passages cost a similar prefill time."""
        words = text.split()
        return self._add(title, [(f"part {j + 1}", " ".join(w)) for j, w in enumerate(windows(words, size, overlap))])

    def add_file(self, path):
        """kb/*.txt: first line '# <title>', then the text (lines starting with '<!--' are comments)."""
        lines = pathlib.Path(path).read_text().splitlines()
        title = lines[0].lstrip("# ").strip()
        return self.add_text(title, "\n".join(l for l in lines[1:] if not l.startswith("<!--")))

    def add_url(self, url):
        """External source: Wikipedia articles through the MediaWiki API (clean text), other pages as HTML."""
        hdr = {"User-Agent": "csed490f-lab7-rag/1.0"}
        m = re.match(r"https?://(\w+)\.wikipedia\.org/wiki/([^#?]+)", url)
        if m:
            lang, page = m.groups()
            r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", headers=hdr, timeout=30, params={
                "action": "query", "prop": "extracts", "explaintext": 1, "format": "json",
                "redirects": 1, "titles": requests.utils.unquote(page)})
            p = next(iter(r.json()["query"]["pages"].values()))
            title, text = f"Wikipedia: {p['title']}", p.get("extract", "")
        else:
            r = requests.get(url, headers=hdr, timeout=30)
            r.raise_for_status()
            parser = _HTMLText()
            parser.feed(r.text)
            text = html.unescape("".join(parser.out))
            tm = re.search(r"<title>(.*?)</title>", r.text, re.S | re.I)
            title = tm.group(1).strip() if tm else url
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return title, self.add_text(title, text)

    def remove(self, doc):
        with self.lock:
            keep = [i for i, c in enumerate(self.chunks) if c["doc"] != doc]
            removed = len(self.chunks) - len(keep)
            self.chunks = [self.chunks[i] for i in keep]
            self.emb = self.emb[keep]
            self._rebuild_bm25()
            self._save()
        return removed

    def docs(self):
        return dict(Counter(c["doc"] for c in self.chunks))

    # ---------------------------------------------------------------- retrieve
    def search(self, query, k=4, mode="hybrid", pool=30):
        """Return (hits, timing).  hits = [{rank, score, doc, loc, text}]."""
        t0 = time.perf_counter()
        with self.lock:
            chunks, emb, bm25 = self.chunks, self.emb, self.bm25
        if not chunks:
            return [], {"embed_ms": 0.0, "search_ms": 0.0}
        q = None
        if mode in ("dense", "hybrid"):
            q = self._embed([query], self.QUERY_PREFIX)[0]
        t1 = time.perf_counter()
        if mode == "dense":
            s = emb @ q
            order = np.argsort(-s)[:k]
            scores = s[order]
        elif mode == "bm25":
            s = bm25.scores(query)
            order = np.argsort(-s)[:k]
            scores = s[order]
        else:  # reciprocal-rank fusion of the two rankings
            rrf = np.zeros(len(chunks), dtype=np.float32)
            for s in (emb @ q, bm25.scores(query)):
                for r, i in enumerate(np.argsort(-s)[:pool]):
                    rrf[i] += 1.0 / (60 + r + 1)
            order = np.argsort(-rrf)[:k]
            scores = rrf[order]
        hits = [{"rank": r + 1, "score": round(float(sc), 4), "doc": chunks[i]["doc"],
                 "loc": chunks[i]["loc"], "text": chunks[i]["text"]} for r, (i, sc) in enumerate(zip(order, scores))]
        t2 = time.perf_counter()
        return hits, {"embed_ms": round((t1 - t0) * 1e3, 1), "search_ms": round((t2 - t1) * 1e3, 1)}

    # ---------------------------------------------------------------- prompt
    @staticmethod
    def label(h):
        """Short source label for the prompt, e.g. 'UEFA Euro 2024, part 3' (long labels cost prefill time)."""
        doc = h["doc"].removeprefix("Wikipedia: ")
        return f"{doc}, {h['loc']}" if h["loc"] else doc

    # Instruction variants compared in eval/ (prompt ablation).  The default is chosen there.
    PROMPTS = {
        "strict": ("Answer the question in one or two sentences using only the passages above. Copy exact "
                   "names, commands, numbers and dates, and cite the passage number like [1]. If the passages "
                   "do not contain the answer, say \"I don't know\"."),
        "noabstain": ("Answer the question in one or two sentences using only the passages above. Copy exact "
                      "names, commands, numbers and dates, and cite the passage number like [1]."),
        "plain": ("Answer the question in one or two sentences using the passages above. "
                  "Copy exact names, commands, numbers and dates from the passages."),
        "soft": ("Answer the question in one or two sentences using the passages above. Copy exact names, "
                 "commands, numbers and dates from the passages. Only if no passage is related to the "
                 "question, say \"I don't know\"."),
    }
    DEFAULT_PROMPT = "plain"     # best accuracy in eval/ (strict abstains more but refuses answerable questions)

    @staticmethod
    def augment(question, hits, variant=None, reverse=False):
        """The user turn sent to the model.  Instructions come after the passages, right before the
        answer.  reverse=True puts the best passage last, nearest to the question."""
        ordered = list(reversed(hits)) if reverse else hits
        ctx = "\n\n".join(f"[{h['rank']}] {KnowledgeBase.label(h)}:\n{h['text']}" for h in ordered)
        instr = KnowledgeBase.PROMPTS[variant or KnowledgeBase.DEFAULT_PROMPT]
        return f"Passages:\n{ctx}\n\nQuestion: {question}\n{instr}"
