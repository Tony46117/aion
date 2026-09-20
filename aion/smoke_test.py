"""Smoke test: engine, introspect, humanizer, learner, fact memory, voice wiring.

Run:  python -m aion.smoke_test
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from aion.engine import Engine
from aion.humanizer import humanize
from aion import introspect
from aion.voice import espeak_available


def main() -> int:
    tmp = Path(tempfile.mkdtemp()) / ".aion-test"
    eng = Engine()
    eng.learner = eng.learner.__class__(mem_dir=tmp)  # isolated memory for the test

    print("== talk mode ==")
    r1 = eng.reply("I keep having dreams about drowning. What do they mean?")
    print(r1[:300], "...\n")

    print("== deep mode ==")
    r2 = eng.reply("I am afraid of my own anger.", deep=True)
    print(r2[:300], "...\n")

    print("== learning check ==")
    assert len(eng.learner.turns) == 2, "learner must record every turn"
    assert (tmp / "memory.jsonl").exists(), "memory must persist"
    r3 = eng.reply("I told you before about the drowning dreams - they came back.")
    mem = eng.learner.relevant_memories("drowning dreams")
    print("recalled memories:", mem)
    assert mem, "learner must recall relevant memories"

    print("== name learning across sessions ==")
    eng.learner.learn("My name is Antony and I live in Nairobi.")
    assert eng.learner.name() == "Antony", f"name must be learned, got {eng.learner.facts}"
    # a *new* learner (i.e. next run of the CLI) must still know him
    eng2 = Engine()
    eng2.learner = eng2.learner.__class__(mem_dir=tmp)
    assert eng2.learner.name() == "Antony", "name must persist across sessions"
    assert eng2.learner.facts.get("home") == "live in Nairobi", eng2.learner.facts
    print("facts after restart:", eng2.learner.facts)
    greeted = eng2.reply("Do you remember me?")
    print("greeting reply:", greeted[:200], "...")
    assert "Antony" in greeted, "aion should greet the returning seeker by name"

    print("== fact resurfacing ==")
    eng2.learner.learn("I am afraid of spiders.")
    hit = eng2.learner.relevant_fact("the spiders were in my dream again")
    print("fact callback:", hit)
    assert hit and "spiders" in hit, "relevant fact must resurface"

    print("== introspective layer ==")
    e = introspect.echo("my grandmother keeps appearing in everything", eng2.rng)
    assert e and "grandmother" in e, f"echo must quote the seeker's word, got {e}"
    print("echo:", e)
    img = introspect.image("dreams", eng2.rng)
    assert len(img) > 30, "image pool must yield original imagery"
    print("image:", img)
    # across several replies, introspective elements must appear
    seen = set()
    for q in [
        "I feel my shadow following me at work",
        "why do I dream of drowning every night",
        "I want to become whole but I keep failing",
        "I am angry at my father and I cannot say it",
        "what is the meaning of my anxiety",
    ] * 3:
        seen.add(eng2.reply(q)[:120])
    joined = " ".join(seen).lower()
    assert len(seen) >= 8, "replies must vary, not repeat the same answer"
    print(f"distinct reply openings across 15 turns: {len(seen)}")

    print("== humanizer standalone ==")
    h = humanize(
        "The dream is the small hidden door in the deepest and most intimate "
        "sanctum of the soul. It is not a disguise. It is a message.",
        rng=None,
    )
    print(h[:200], "...\n")

    print("== stats ==")
    print(eng2.stats())

    print("\nvoice available:", espeak_available())
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
