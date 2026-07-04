import os
import re
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")


def parse_year_month(filename):
    m = FILENAME_RE.search(filename)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def naive_chunks(text, chunk_chars=1000):
    """
    BASELINE: fixed-size character chunking, no overlap, only a light
    paragraph-boundary nudge. This is intentionally 'quick and dirty' -
    it will regularly slice a table in half.
    """
    paras = text.split("\n\n")
    chunks, current = [], ""
    for p in paras:
        if len(current) + len(p) + 2 <= chunk_chars:
            current = current + "\n\n" + p if current else p
        else:
            if current:
                chunks.append(current)
            # if a single paragraph is itself too long, hard-slice it
            while len(p) > chunk_chars:
                chunks.append(p[:chunk_chars])
                p = p[chunk_chars:]
            current = p
    if current:
        chunks.append(current)
    return chunks


def token_overlap_chunks(text, chunk_tokens=512, overlap_tokens=75):
    """
    ENGINEERED: token-based sliding window with ~15% overlap, so a
    fact split across a chunk boundary in one window still appears
    whole in the neighboring window.
    """
    tokens = ENC.encode(text)
    chunks = []
    start = 0
    step = chunk_tokens - overlap_tokens
    while start < len(tokens):
        window = tokens[start:start + chunk_tokens]
        chunks.append(ENC.decode(window))
        if start + chunk_tokens >= len(tokens):
            break
        start += step
    return chunks


def load_bulletins(bulletins_dir):
    """Returns list of dicts: {filename, year, month, text}. Walks recursively
    since hf_hub_download preserves the repo's nested folder structure."""
    out = []
    for root, _, files in os.walk(bulletins_dir):
        for fname in sorted(files):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(root, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            year, month = parse_year_month(fname)
            if year is None:
                continue
            out.append({"filename": fname, "year": year, "month": month, "text": text})
    return out
