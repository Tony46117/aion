"""Extract and clean the raw text of the training books into a single corpus.

Looks for PDFs (and .txt files) in ``aion/data/books/`` and writes a cleaned,
sentence-oriented corpus to ``aion/data/corpus.txt``.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
BOOKS_DIR = PKG_DIR / "data" / "books"
CORPUS_PATH = PKG_DIR / "data" / "corpus.txt"

FOOTER_NOISE = re.compile(r"_?OceanofPDF\.com_?", re.IGNORECASE)
PAGE_NUMBERS = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
MULTI_BLANK = re.compile(r"\n{3,}")


def _clean_text(raw: str) -> str:
    """Normalize a book's raw text into flowing prose."""
    # Normalize unicode punctuation so the model sees consistent tokens.
    raw = unicodedata.normalize("NFKC", raw)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2014": " -- ", "\u2013": "-", "\u2026": "...",
        "\ufb01": "fi", "\ufb02": "fl", "\u00a0": " ",
    }
    for bad, good in replacements.items():
        raw = raw.replace(bad, good)

    raw = FOOTER_NOISE.sub("", raw)
    raw = PAGE_NUMBERS.sub("", raw)

    # Join lines within a paragraph; keep blank lines as paragraph breaks.
    lines = [ln.rstrip() for ln in raw.splitlines()]
    out: list[str] = []
    buf: list[str] = []
    for ln in lines:
        if not ln.strip():
            if buf:
                out.append(" ".join(buf))
                buf = []
        else:
            buf.append(ln.strip())
    if buf:
        out.append(" ".join(buf))

    text = "\n\n".join(out)
    text = re.sub(r"\s+", " ", text)          # collapse intra-paragraph space
    text = re.sub(r"\n{2,}", "\n", text)
    text = MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def build_corpus() -> Path:
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    sources: list[str] = []

    for book in sorted(BOOKS_DIR.iterdir()):
        if book.name.startswith("."):
            continue
        if book.suffix.lower() == ".pdf":
            print(f"  extracting {book.name} ...")
            parts.append(_clean_text(extract_pdf(book)))
            sources.append(book.name)
        elif book.suffix.lower() in {".txt", ".md"}:
            print(f"  reading {book.name} ...")
            parts.append(_clean_text(book.read_text(encoding="utf-8", errors="ignore")))
            sources.append(book.name)

    if not parts:
        raise SystemExit(
            f"No books found in {BOOKS_DIR}. Drop Jungian PDFs there and re-run."
        )

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus = ("\n\n" + "=" * 60 + "\n\n").join(parts)
    CORPUS_PATH.write_text(corpus, encoding="utf-8")

    n_words = len(corpus.split())
    print(f"corpus built: {CORPUS_PATH}")
    print(f"  sources : {len(sources)} -> {', '.join(sources)[:120]}")
    print(f"  size    : {len(corpus):,} chars / {n_words:,} words")
    return CORPUS_PATH


if __name__ == "__main__":
    build_corpus()
