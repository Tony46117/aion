"""The aion engine: a Jung persona built from the trained heads.

Retrieval (TF-IDF over the corpus + learned memory), a topic classifier
(the teachings head), a char-CNN style scorer (the style head), and a
composer that assembles answers the way Jung spoke: mirror first, image
second, question last.
"""
from __future__ import annotations

import json
import pickle
import random
import re
from pathlib import Path

from . import introspect
from .humanizer import humanize
from .learner import Learner

PKG_DIR = Path(__file__).resolve().parent
MODELS_DIR = PKG_DIR / "models"

TOPICS = [
    "shadow", "anima_animus", "dreams", "individuation", "god_religion",
    "psyche", "therapy", "alchemy", "types",
]

# Topical anchors the composer can lean on once a topic is classified.
ANCHORS: dict[str, list[str]] = {
    "shadow": [
        "That which you refuse to see in yourself does not disappear; it waits, and it chooses your moments for you.",
        "Everyone carries a shadow, and the less it is embodied in the individual's conscious life, the blacker and denser it is.",
        "What you despise in the other man is usually the unlived life of your own soul.",
    ],
    "anima_animus": [
        "The anima is not the woman outside; she is the image of life itself that stirs within a man, and she has moods you did not order.",
        "Every man carries a woman within him, and every woman a man; they speak in the voice of feeling when you least expect it.",
        "The encounter with the anima is the encounter with your own soul, and she will not be managed.",
    ],
    "dreams": [
        "The dream is the small hidden door in the deepest and most intimate sanctum of the soul.",
        "A dream that is not understood remains a letter that has been posted but never opened.",
        "The dream does not conceal; it reveals, in the only language the unconscious trusts, which is image.",
    ],
    "individuation": [
        "Individuation means becoming an in-dividual, and in so far as individuality embraces our innermost, last, and incomparable uniqueness, it also implies becoming one's own self.",
        "You do not become whole by polishing one half; the aim is the reconciliation of the opposites, however uncomfortable that proves.",
        "The privilege of a lifetime is to become who you truly are, and it costs everything you pretend to be.",
    ],
    "god_religion": [
        "The gods have not died; they have gone underground and now call themselves complexes, addictions, and ideologies.",
        "Religion is a careful and scrupulous observation of what Rudolf Otto aptly termed the numinosum, and it meets you where your carefully built certainties end.",
        "Among all my patients in the second half of life, every one fell ill because he had lost that which the living religions of every age have given their followers.",
    ],
    "psyche": [
        "The psyche is the world's pivot: without it there would be no world at all, for the world is as it is only in so far as it is known.",
        "As far as we can discern, the sole purpose of human existence is to kindle a light in the darkness of mere being.",
        "The unconscious is not a cellar you can tidy once; it is a living process that speaks to you continuously, if you will listen.",
    ],
    "therapy": [
        "The doctor is effective only when he himself is affected; only the wounded physician heals.",
        "The patient is there to be met, not repaired; the cure begins when two psychic systems touch and both are changed.",
        "Learn your theories as well as you can, but scorn them at the contact with the living man.",
    ],
    "alchemy": [
        "The alchemists projected into their vessels the whole drama of the psyche's transformation, and called the substance by many names because it was one process in many disguises.",
        "What the old masters sought in the retort was not gold but the medicine of the soul, and their language was projection because they did not yet know it was themselves.",
    ],
    "types": [
        "The thinking type and the feeling type imprison each other in equal measure, for each mistakes his own one-sidedness for the whole truth.",
        "Every type carries its inferior function like a shadow, and there, precisely there, the growth is waiting.",
    ],
}

OPENERS = [
    "You bring this to me as a question, but it is also a symptom.",
    "I hear in your words something that is asking to be spoken.",
    "This did not come to you by accident.",
    "What you describe has a familiar shape; the psyche repeats what it must.",
    "Let us stay with this a moment instead of rushing past it.",
    "You ask as if the answer were outside you.",
]

CLOSERS = [
    "What would it mean if this were not your enemy, but your messenger?",
    "Where in your own life does this figure wish to be lived?",
    "Ask instead: what is this asking of you, not what does it mean.",
    "The question is not whether it is true, but whether it is yours.",
    "What do you do, precisely, when this feeling takes the floor?",
]


def _load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


class Engine:
    """Loads the trained artifacts and composes Jungian answers."""

    def __init__(self) -> None:
        self.rng = random.Random()
        self.teachings = None
        self.style_model = None
        self.style_meta: dict = {}
        self.kb: dict = {"paragraphs": [], "sentences": []}
        self.learner = Learner()
        self._torch_ready = False
        self._session_turn = 0
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if (MODELS_DIR / "teachings.joblib").exists():
            with open(MODELS_DIR / "teachings.joblib", "rb") as f:
                self.teachings = pickle.load(f)
        if (MODELS_DIR / "knowledge.json").exists():
            self.kb = _load_json(MODELS_DIR / "knowledge.json")
        if (MODELS_DIR / "style_meta.json").exists():
            self.style_meta = _load_json(MODELS_DIR / "style_meta.json")
        try:
            import torch
            import torch.nn as nn

            if (MODELS_DIR / "style_cnn.pth").exists():
                chars = self.style_meta.get("chars", "")
                max_len = self.style_meta.get("max_len", 160)

                class _CharCNN(nn.Module):
                    # must mirror aion.train.CharCNN exactly (channels=72)
                    def __init__(self, vocab: int, embed=48, channels=72):
                        super().__init__()
                        self.emb = nn.Embedding(vocab + 1, embed, padding_idx=0)
                        self.convs = nn.ModuleList(
                            nn.Conv1d(embed, channels, k, padding=k // 2)
                            for k in (2, 3, 4, 5)
                        )
                        self.drop = nn.Dropout(0.3)
                        self.fc = nn.Linear(channels * 4, 1)

                    def forward(self, x):
                        e = self.emb(x).transpose(1, 2)
                        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
                        return self.fc(self.drop(torch.cat(feats, dim=1))).squeeze(-1)

                self._cnn_cls = _CharCNN
                self._torch = torch
                self.style_model = _CharCNN(len(chars))
                self.style_model.load_state_dict(
                    torch.load(MODELS_DIR / "style_cnn.pth", map_location="cpu")
                )
                self.style_model.eval()
                self._encode = self._make_encoder(chars, max_len)
                self._torch_ready = True
        except Exception:
            self.style_model = None
            self._torch_ready = False

    @staticmethod
    def _make_encoder(chars: str, max_len: int):
        c2i = {c: i + 1 for i, c in enumerate(chars)}

        def encode(s: str):
            import torch
            s = s.lower()[:max_len]
            ids = [c2i.get(c, 0) for c in s]
            ids += [0] * max(0, max_len - len(ids))
            return torch.tensor([ids], dtype=torch.long)

        return encode

    # ------------------------------------------------------------------
    def _style_score(self, text: str) -> float:
        if not self._torch_ready:
            # graceful fallback: prefer longer, image-rich, non-list prose
            import math
            words = text.split()
            if not words:
                return -1e9
            commas = text.count(",") / max(1, len(words))
            return math.log(len(words) + 1) + commas
        with self._torch.no_grad():
            return float(self.style_model(self._encode(text)))

    def _classify(self, text: str) -> str:
        if self.teachings is None:
            return "psyche"
        try:
            return self.teachings.predict([text])[0]
        except Exception:
            return "psyche"

    def _retrieve(self, query: str, k: int = 6) -> list[str]:
        """Most relevant corpus paragraphs + remembered conversation lines."""
        pool: list[str] = list(self.kb.get("paragraphs", []))

        remembered = self.learner.relevant_memories(query, k=4)
        pool.extend(remembered)

        if self.teachings is not None and pool:
            try:
                vec = self.teachings.named_steps["tfidf"].transform(
                    [query] + pool
                )
                import numpy as np
                sims = (vec @ vec[0].T).toarray().ravel()[1:]
                order = np.argsort(-sims)[:k]
                picked = [pool[int(i)] for i in order if sims[int(i)] > 0.05]
                if picked:
                    return picked
            except Exception:
                pass
        return pool[:k]

    # ------------------------------------------------------------------
    def _topics_in(self, text: str) -> list[str]:
        low = text.lower()
        hits = []
        for t in TOPICS:
            keys = {
                "shadow": ["shadow", "dark", "enemy", "hate", "blame"],
                "anima_animus": ["anima", "animus", "love", "woman", "man", "marriage"],
                "dreams": ["dream", "nightmare", "sleep", "vision"],
                "individuation": ["purpose", "whole", "myself", "who am i", "become"],
                "god_religion": ["god", "religion", "meaning of life", "sacred", "soul"],
                "psyche": ["unconscious", "psyche", "mind", "soul", "conscious"],
                "therapy": ["therapy", "depress", "anxious", "anxiety", "sick", "hurt", "pain"],
                "alchemy": ["alchemy", "transformation", "gold", "stone"],
                "types": ["introvert", "extrovert", "extravert", "thinking", "feeling", "type"],
            }.get(t, [])
            if any(k in low for k in keys):
                hits.append(t)
        if not hits:
            hits = [self._classify(low)]
        return hits[:2]

    def _core(self, user_text: str, deep: bool) -> str:
        """Compose a reply that reacts like a person, not an index.

        Order: mirror the seeker's own word, admit an inner state, offer the
        teaching, add an *original* image, and only occasionally surface a
        corpus line - framed as something remembered, not recited.
        """
        topics = self._topics_in(user_text)
        retrieved = self._retrieve(user_text)

        # Harvest a strong corpus sentence only sometimes; deep sessions allow more.
        quote = None
        if deep or self.rng.random() < 0.45:
            for p in retrieved:
                sents = re.split(r"(?<=[.!?])\s+", p)
                long_ones = [s for s in sents if 60 <= len(s) <= 260]
                if long_ones:
                    scored = max(long_ones, key=self._style_score)
                    if self._style_score(scored) > 0:
                        quote = scored
                        break

        main = topics[0]
        parts: list[str] = []

        # 1. mirror: hold one of the seeker's own words up to the light
        if not deep and self.rng.random() < 0.55:
            echo = introspect.echo(user_text, self.rng)
            if echo:
                parts.append(echo)

        # 2. introspection: aion owns an inner reaction
        if self.rng.random() < (0.35 if deep else 0.50):
            parts.append(introspect.self_note(self.rng))

        # 3. the teaching - one anchor in talk mode, richer in deep mode
        for t in (topics if deep else topics[:1]):
            parts.append(self.rng.choice(ANCHORS.get(t, ANCHORS["psyche"])))

        # 4. an original image, composed for the moment (never a quotation)
        if self.rng.random() < (0.70 if deep else 0.65):
            parts.append(introspect.image(main, self.rng))

        # 5. corpus quote, framed as remembered speech: rare in talk, common in deep
        if quote and quote.lower() not in " ".join(parts).lower():
            if self.rng.random() < (0.90 if deep else 0.25):
                parts.append(f'{introspect.quote_frame(self.rng)} "{quote}"')

        # 6. awareness of the conversation itself (deep sessions)
        if deep and self.rng.random() < 0.25:
            parts.append(introspect.dialogue_note(self.rng))

        # 7. the question that turns it back on the seeker
        parts.append(self.rng.choice(CLOSERS))
        return " ".join(parts)

    # ------------------------------------------------------------------
    def reply(self, user_text: str, deep: bool = False, user: str = "seeker") -> str:
        core = self._core(user_text, deep)

        # greet a remembered seeker by name once per session
        first_turn = self._session_turn == 0
        self._session_turn += 1
        name = self.learner.name()
        if first_turn and name and self.rng.random() < 0.75 and core:
            core = f"So, {name} - {core[0].lower()}{core[1:]}"

        # resurface a remembered fact when the seeker's words touch it
        fact = self.learner.relevant_fact(user_text)
        if fact and self.rng.random() < 0.8:
            core = f"{core} {fact}"

        moods = self.learner.mood_profile()
        text = humanize(core, rng=self.rng, mood=moods, deep=deep)
        self.learner.record_turn(user_text, text, deep=deep)
        return text

    def stats(self) -> dict:
        return {
            "knowledge_base": len(self.kb.get("paragraphs", [])),
            "style_model": "char-CNN" if self._torch_ready else "fallback heuristics",
            "teachings": "tf-idf + logistic" if self.teachings else "not loaded",
            **self.learner.stats(),
        }
