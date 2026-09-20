"""Learner: aion's memory. Learns from every conversation, forever.

Every turn you exchange with aion is folded back into its memory:

- turns are persisted to ``~/.aion/memory.jsonl`` (durable across sessions)
- TF-IDF vectors are *incrementally refit* over remembered turns so future
  retrieval pulls in what you have actually talked about, not just the books
- a mood/topic profile accumulates (which Jungian themes you keep bringing,
  whether your questions skew anxious, angry, seeking...)
- what aion "remembers" bleeds back into answers through retrieval and the
  humanizer's mood flavor ("You have brought dreams to me before...")

The design is deliberately simple and honest: no pretending to be a large
language model, but a real statistical learner that measurably changes its
retrieval and tone as you talk with it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MEM_DIR = Path.home() / ".aion"
MEMORY_FILE = MEM_DIR / "memory.jsonl"
PROFILE_FILE = MEM_DIR / "profile.json"

TOPIC_WORDS: dict[str, tuple[str, ...]] = {
    "your shadow": ("shadow", "dark", "hate", "enemy", "blame", "angry"),
    "dreams": ("dream", "nightmare", "sleep", "vision", "night"),
    "the anima": ("anima", "animus", "love", "woman", "man", "marriage", "lover"),
    "individuation": ("purpose", "whole", "become", "myself", "who am i", "career"),
    "the god-image": ("god", "religion", "sacred", "church", "prayer", "faith"),
    "the unconscious": ("unconscious", "psyche", "mind", "soul", "conscious"),
    "suffering": ("pain", "hurt", "depress", "anxiety", "anxious", "afraid", "fear", "sad"),
}

VIBES: dict[str, tuple[str, ...]] = {
    "anxious": ("afraid", "anxious", "worry", "panic", "scared", "nervous"),
    "sorrowful": ("sad", "grief", "loss", "cry", "lonely", "empty"),
    "searching": ("meaning", "why", "purpose", "who am i", "lost", "confused"),
    "struggling": ("angry", "hate", "fight", "battle", "cannot", "stuck"),
}


class Learner:
    """Persists conversation memory and refits retrieval over it."""

    def __init__(self, mem_dir: Path | None = None) -> None:
        self.mem_dir = mem_dir or MEM_DIR
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.mem_dir / "memory.jsonl"
        self.profile_file = self.mem_dir / "profile.json"

        self.turns: list[dict] = []
        self._load()
        self._refit()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.memory_file.exists():
            try:
                for line in self.memory_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        self.turns.append(json.loads(line))
            except Exception:
                self.turns = []
        self.profile: dict = {}
        if self.profile_file.exists():
            try:
                self.profile = json.loads(self.profile_file.read_text(encoding="utf-8"))
            except Exception:
                self.profile = {}

    def _persist(self) -> None:
        try:
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.turns[-1], ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def record_turn(self, user_text: str, aion_text: str, deep: bool = False) -> None:
        entry = {
            "t": time.strftime("%Y-%m-%d %H:%M:%S"),
            "deep": deep,
            "user": user_text,
            "aion": aion_text,
        }
        self.turns.append(entry)
        self._persist()
        self._update_profile(user_text)
        # refit periodically, not every keystroke
        if len(self.turns) % 5 == 0 or len(self.turns) < 10:
            self._refit()

    def _update_profile(self, user_text: str) -> None:
        low = " " + user_text.lower() + " "
        prof = self.profile
        # topics
        topics = prof.setdefault("topics", {})
        for topic, words in TOPIC_WORDS.items():
            if any(w in low for w in words):
                topics[topic] = topics.get(topic, 0) + 1
        prof["topics"] = topics
        # vibes
        vibes = prof.setdefault("vibes", {})
        for vibe, words in VIBES.items():
            if any(w in low for w in words):
                vibes[vibe] = vibes.get(vibe, 0) + 1
        prof["vibes"] = vibes
        prof["n_user_turns"] = prof.get("n_user_turns", 0) + 1
        prof["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_profile()

    def _save_profile(self) -> None:
        try:
            self.profile_file.write_text(
                json.dumps(self.profile, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def _refit(self) -> None:
        """Rebuild the memory TF-IDF index over remembered turns."""
        docs = [f"{t.get('user', '')} {t.get('aion', '')}" for t in self.turns]
        docs = [d for d in docs if d.strip()]
        if not docs:
            self._mem_vec = None
            self._mem_docs = []
            return
        try:
            self._mem_vec = TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=40000
            )
            self._mem_mat = self._mem_vec.fit_transform(docs)
            self._mem_docs = docs
        except Exception:
            self._mem_vec = None
            self._mem_docs = []

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------
    def relevant_memories(self, query: str, k: int = 3) -> list[str]:
        """Remembered exchanges relevant to the query, phrased as memory."""
        if self._mem_vec is None or not self.turns:
            return []
        try:
            import numpy as np
            q = self._mem_vec.transform([query])
            sims = cosine_similarity(q, self._mem_mat).ravel()
            idx = np.argsort(-sims)[:k]
            out = []
            for i in idx:
                if sims[int(i)] > 0.12:
                    t = self.turns[int(i)]
                    what = t.get("user", "")[:160]
                    out.append(f"You once told me: \"{what}\"")
            return out
        except Exception:
            return []

    def mood_profile(self) -> dict:
        topics = self.profile.get("topics", {})
        vibes = self.profile.get("vibes", {})
        fav = max(topics, key=topics.get) if topics else None
        vibe = max(vibes, key=vibes.get) if vibes else None
        return {
            "favorite_topic": fav,
            "vibe": vibe,
            "n_user_turns": self.profile.get("n_user_turns", 0),
            "n_sessions": self.profile.get("n_user_turns", 0) // 20,
        }

    def stats(self) -> dict:
        topics = self.profile.get("topics", {})
        vibes = self.profile.get("vibes", {})
        return {
            "remembered_turns": len(self.turns),
            "topics_learned": topics,
            "vibes_learned": vibes,
            "memory_file": str(self.memory_file),
        }

    def forget(self) -> None:
        self.turns = []
        self.profile = {}
        try:
            self.memory_file.unlink(missing_ok=True)
            self.profile_file.unlink(missing_ok=True)
        except Exception:
            pass
        self._refit()
