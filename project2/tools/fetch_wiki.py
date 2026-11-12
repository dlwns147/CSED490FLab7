"""Build the knowledge base: snapshot Wikipedia articles (plain text, pinned revision) into kb/*.txt.

    python tools/fetch_wiki.py            # the articles listed below
Wikipedia text is licensed CC BY-SA 4.0; each file records its source URL and revision id.
"""
import pathlib
import re
import time

import requests

ARTICLES = [   # mostly 2024-2025 topics: after Llama 3.2's training cutoff (Dec 2023), so the model cannot know them
    "DeepSeek", "Llama (language model)", "Gemini (language model)", "Blackwell (microarchitecture)", "Apple M4",
    "UEFA Euro 2024", "2024 Summer Olympics", "Solar eclipse of April 8, 2024",
]
OUT = pathlib.Path(__file__).resolve().parent.parent / "kb"
H = {"User-Agent": "csed490f-lab7-rag/1.0 (course project)"}


def fetch(title, tries=6):
    for i in range(tries):                             # Wikipedia rate-limits bursts (HTTP 429): back off and retry
        r = _get(title)
        if r.status_code != 429:
            break
        time.sleep(int(r.headers.get("Retry-After", 0)) or 30 * (i + 1))
    r.raise_for_status()
    return _parse(r)


def _get(title):
    return requests.get("https://en.wikipedia.org/w/api.php", headers=H, timeout=30, params={
        "action": "query", "prop": "extracts|revisions", "explaintext": 1, "redirects": 1, "format": "json",
        "rvprop": "ids|timestamp", "titles": title})


def _parse(r):
    p = next(iter(r.json()["query"]["pages"].values()))
    rev = p["revisions"][0]
    text = re.sub(r"\n\s*\n+", "\n\n", p["extract"])
    for cut in ("\n== References ==", "\n== See also ==", "\n== Notes ==", "\n== External links =="):
        text = text.split(cut)[0]                      # drop trailing reference sections
    url = f"https://en.wikipedia.org/w/index.php?title={p['title'].replace(' ', '_')}&oldid={rev['revid']}"
    return p["title"], url, rev["timestamp"], text.strip()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for t in ARTICLES:
        title, url, ts, text = fetch(t)
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        (OUT / f"{slug}.txt").write_text(f"# Wikipedia: {title}\n<!-- {url} ({ts}), CC BY-SA 4.0 -->\n\n{text}\n")
        print(f"{len(text.split()):6d} words  {title}")
        time.sleep(2)
