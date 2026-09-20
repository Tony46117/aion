"""Train the aion persona on the Jungian corpus.

Two heads are trained for 100+ epochs:

1. **Teachings head** — TF-IDF (word 1-2 grams) + LogisticRegression over
   corpus paragraphs labeled by a keyword router (shadow, anima, dreams,
   individuation, God/religion, psyche, therapy, self/individuation...).
   At inference this retrieves the most relevant teaching for a user's topic.

2. **Style head** — a small character-level CNN (PyTorch) trained on Jung's
   own sentences to score how *Jungian* a candidate sentence sounds. The
   engine uses it to rank and select generated variants.

Artifacts are written to ``aion/models/`` and committed with the repo, so the
CLI runs without retraining.
"""
from __future__ import annotations

import json
import math
import pickle
import random
import re
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
MODELS_DIR = PKG_DIR / "models"
CORPUS_PATH = DATA_DIR / "corpus.txt"

# --------------------------------------------------------------------------
# Corpus handling
# --------------------------------------------------------------------------

ABBREV = ("dr.", "mr.", "mrs.", "st.", "cf.", "e.g.", "i.e.", "no.", "vol.", "p.", "pp.", "cf")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\u2018'\"])", text)
    out = []
    for s in parts:
        s = s.strip()
        low = s.lower()
        # merge fragments created by abbreviations
        if out and (len(s) < 25 or low.startswith(ABBREV)):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return [s for s in out if 30 <= len(s) <= 600]


def split_paragraphs(text: str) -> list[str]:
    """Paragraph-sized passage units.

    PDF-extracted corpora often lose paragraph breaks, so we split on
    newlines first and, if that yields too little, fall back to grouping
    sentences into blocks of ~4 (a natural retrieval unit).
    """
    paras = [p.strip() for p in re.split(r"\n", text) if p.strip()]
    paras = [p for p in paras if 80 <= len(p) <= 1200]
    if len(paras) < 50:  # corpus lost its paragraph breaks -> group sentences
        sents = split_sentences(text)
        paras = []
        for i in range(0, len(sents) - 3, 4):
            block = " ".join(sents[i:i + 4])
            if 80 <= len(block) <= 1200:
                paras.append(block)
    return paras


# --------------------------------------------------------------------------
# Teachings head
# --------------------------------------------------------------------------

TOPIC_KEYS: dict[str, tuple[str, ...]] = {
    "shadow": (
        "shadow", "dark side", "the dark", "repressed", "inferior function",
        "persona", "mask", "projection", "projected", "enemy", "blame",
    ),
    "anima_animus": (
        "anima", "animus", "contrasexual", "inner woman", "inner man",
        "eros", "logos", "feminine", "masculine", "love", "woman", "man and",
    ),
    "dreams": (
        "dream", "dreams", "dreamt", "night", "sleep", "vision", "fantasy",
        "imagination", "active imagination", "symbol", "symbols", "image",
    ),
    "individuation": (
        "individuation", "self", "selfhood", "wholeness", "becoming whole",
        "mandala", "quaternity", "circle", "centre", "center", "the process",
    ),
    "god_religion": (
        "god", "gods", "christ", "religion", "church", "creed", "dogma",
        "holy", "sacred", "numinous", "devil", "evil", "Job", "Trinity",
    ),
    "psyche": (
        "psyche", "psychic", "soul", "spirit", "unconscious", "conscious",
        "consciousness", "collective unconscious", "archetype", "archetypes",
    ),
    "therapy": (
        "patient", "patients", "analyst", "analysis", "doctor", "treatment",
        "therapy", "neurosis", "neurotic", "symptom", "suffering", "cure",
    ),
    "alchemy": (
        "alchemy", "alchemical", "mercurius", "lapis", "philosopher's stone",
        "gold", " Prima", "opus", "quicksilver", "reactive substance",
    ),
    "types": (
        "extravert", "introvert", "extraversion", "introversion", "thinking",
        "feeling", "sensation", "intuition", "function", "attitude", "type",
    ),
}


def label_for(paragraph: str) -> str | None:
    low = " " + paragraph.lower() + " "
    best, hits = None, 0
    for topic, keys in TOPIC_KEYS.items():
        n = sum(low.count(k) for k in keys)
        if n > hits:
            best, hits = topic, n
    return best


def train_teachings(paragraphs: list[str], epochs: int, report: dict) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline

    labeled = [(p, label_for(p)) for p in paragraphs]
    labeled = [(p, t) for p, t in labeled if t]
    texts = [p for p, _ in labeled]
    labels = [t for _, t in labeled]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.85,
            sublinear_tf=True, max_features=80000,
        )),
        ("clf", SGDClassifier(loss="log_loss", alpha=1e-6, random_state=0)),
    ])

    rng = random.Random(42)
    n = len(texts)
    print(f"  teachings: {n:,} labeled paragraphs, {epochs} epochs")
    pipe.fit(texts, labels)  # warm start
    for ep in range(1, epochs + 1):
        idx = list(range(n))
        rng.shuffle(idx)
        # partial fits over minibatches keep the 100+ epoch loop honest
        B = 1024
        for i in range(0, n, B):
            batch = [texts[j] for j in idx[i:i + B]]
            blab = [labels[j] for j in idx[i:i + B]]
            pipe.named_steps["clf"].partial_fit(
                pipe.named_steps["tfidf"].transform(batch), blab,
                classes=pipe.classes_,
            )
        if ep % max(1, epochs // 10) == 0 or ep == epochs:
            acc = pipe.score(texts, labels)
            print(f"    epoch {ep:>4}/{epochs}  train-acc={acc:.3f}")
            report["teachings_epochs"].append({"epoch": ep, "train_acc": round(float(acc), 4)})

    with open(MODELS_DIR / "teachings.joblib", "wb") as f:
        pickle.dump(pipe, f)


# --------------------------------------------------------------------------
# Style head (char-CNN)
# --------------------------------------------------------------------------

CHARS = " \nabcdefghijklmnopqrstuvwxyz0123456789.,;:!?'-\"()[]"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
MAX_LEN = 112
CKPT_EVERY = 12


def encode_text(s: str) -> list[int]:
    s = s.lower()[:MAX_LEN]
    return [CHAR2IDX.get(c, 0) for c in s] + [0] * max(0, MAX_LEN - len(s))


def train_style(sentences: list[str], epochs: int, report: dict, resume: bool = False) -> None:
    import numpy as np
    import torch
    import torch.nn as nn

    torch.set_num_threads(max(1, (torch.get_num_threads() or 2) * 2))
    torch.manual_seed(42)
    random.seed(42)

    class CharCNN(nn.Module):
        def __init__(self, vocab: int, embed: int = 48, channels: int = 72):
            super().__init__()
            self.emb = nn.Embedding(vocab + 1, embed, padding_idx=0)
            self.convs = nn.ModuleList(
                nn.Conv1d(embed, channels, k, padding=k // 2) for k in (2, 3, 4, 5)
            )
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(channels * 4, 1)

        def forward(self, x):                      # x: (B, L)
            e = self.emb(x).transpose(1, 2)        # (B, E, L)
            feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
            return self.fc(self.drop(torch.cat(feats, dim=1))).squeeze(-1)

    # Style pairs: Jung's sentences (positive) vs. same words shuffled (negative)
    rng = random.Random(7)
    pos = [s for s in sentences if 25 <= len(s) <= 220]
    neg = []
    for s in pos:
        words = s.split()
        if len(words) > 5:
            mid = words[1:-1]
            rng.shuffle(mid)
            neg.append(" ".join([words[0]] + mid + [words[-1]]))
    pos, neg = pos[:9000], neg[:9000]

    X = torch.tensor([encode_text(s) for s in pos + neg], dtype=torch.long)
    y = torch.tensor([1.0] * len(pos) + [0.0] * len(neg))

    model = CharCNN(len(CHARS))
    start_ep = 0
    ckpt_path = MODELS_DIR / "style_ckpt.pth"
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        start_ep = ck["epoch"]
        print(f"  resumed style head from epoch {start_ep}")
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if start_ep:
        for _ in range(start_ep):
            sched.step()
    lossf = nn.BCEWithLogitsLoss()

    B = 256
    n = X.shape[0]
    print(f"  style: {n:,} samples (char-CNN), {epochs} epochs on {torch.get_num_threads()} threads")
    for ep in range(start_ep + 1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot, seen = 0.0, 0
        for i in range(0, n, B):
            idx = perm[i:i + B]
            opt.zero_grad()
            out = model(X[idx])
            loss = lossf(out, y[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
            seen += len(idx)
        sched.step()
        if ep % CKPT_EVERY == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                correct = ((model(X) > 0) == (y > 0)).float().mean().item()
            print(f"    epoch {ep:>4}/{epochs}  loss={tot/seen:.4f}  train-acc={correct:.3f}", flush=True)
            report["style_epochs"].append({"epoch": ep, "loss": round(tot / seen, 4), "train_acc": round(correct, 4)})
            torch.save({"model": model.state_dict(), "epoch": ep}, ckpt_path)  # checkpoint

    torch.save(model.state_dict(), MODELS_DIR / "style_cnn.pth")
    with open(MODELS_DIR / "style_meta.json", "w") as f:
        json.dump({"chars": CHARS, "max_len": MAX_LEN}, f)
    ckpt_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Knowledge base for the engine
# --------------------------------------------------------------------------

def build_knowledge(paragraphs: list[str], sentences: list[str]) -> None:
    kb = {
        "paragraphs": paragraphs[:6000],
        "sentences": sentences[:30000],
        "n_chars": sum(len(p) for p in paragraphs),
    }
    with open(MODELS_DIR / "knowledge.json", "w") as f:
        json.dump(kb, f, ensure_ascii=False)


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------

def main(epochs: int = 120, quick: bool = False, style_only: bool = False, resume: bool = False) -> None:
    t0 = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not CORPUS_PATH.exists():
        from .ingest import build_corpus
        build_corpus()

    text = CORPUS_PATH.read_text(encoding="utf-8", errors="ignore")
    print(f"corpus: {len(text):,} chars")

    if quick:
        paragraphs = split_paragraphs(text)[:1500]
        sentences = split_sentences(text)[:4000]
        epochs = min(epochs, 12)
    else:
        paragraphs = split_paragraphs(text)
        sentences = split_sentences(text)

    report: dict = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs_requested": epochs,
        "teachings_epochs": [],
        "style_epochs": [],
        "corpus_chars": len(text),
        "n_paragraphs": len(paragraphs),
        "n_sentences": len(sentences),
    }

    if style_only:
        print(f"training style head ({epochs} epochs) [style-only mode]")
        train_style(sentences, epochs, report, resume=resume)
        report["minutes"] = round((time.time() - t0) / 60, 1)
        with open(MODELS_DIR / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"done in {report['minutes']} min -> {MODELS_DIR}")
        return

    print(f"training teachings head ({epochs} epochs) ...")
    train_teachings(paragraphs, epochs, report)

    print(f"training style head ({epochs} epochs) ...")
    train_style(sentences, epochs, report, resume=resume)

    build_knowledge(paragraphs, sentences)

    report["minutes"] = round((time.time() - t0) / 60, 1)
    with open(MODELS_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"done in {report['minutes']} min -> {MODELS_DIR}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="train the aion persona")
    ap.add_argument("--epochs", type=int, default=120, help="training epochs (100+ for the real run)")
    ap.add_argument("--quick", action="store_true", help="small smoke run")
    ap.add_argument("--style-only", action="store_true", help="(re)train only the style head")
    ap.add_argument("--resume", action="store_true", help="resume style head from checkpoint")
    a = ap.parse_args()
    main(epochs=a.epochs, quick=a.quick, style_only=a.style_only, resume=a.resume)
