# aion — a Jungian AI for the terminal

> "One does not become enlightened by imagining figures of light, but by making the darkness conscious."

**aion** is a terminal CLI chatbot that speaks as a representation of C.G. Jung. It is
trained (small-scale, on-device) on a corpus of Jungian books, runs fully offline in
pure Python, and grows from every conversation you have with it.

## Architecture

| Module | Purpose |
|---|---|
| `cli.py` | Terminal app (Rich + prompt_toolkit) — the opencode-style UI |
| `engine.py` | Jung persona brain: TF-IDF retrieval + char-ngram logistic classifier + generative composer |
| `introspect.py` | The inner life: echoes the seeker's own words, self-reflection, original imagery — speaks *to* you, not from an index |
| `humanizer.py` | Humanizes engine output: contractions, hedging, rhythm, Jungian interjections |
| `learner.py` | Learns from every conversation; persists memory to `~/.aion/` so aion gets smarter over time |
| `voice.py` | Jung's voice: espeak-ng TTS with a lowered pitch/slowed cadence (bilingual German accent voice) |
| `train.py` | Trains the persona on the corpus (TF-IDF + logistic head on teachings, char-CNN on style), 100+ epochs |
| `ingest.py` | Extracts and cleans the raw text of the PDF books into `data/corpus.txt` |
| `train_cli.py` | Thin wrapper to run training from the installed `aion` command |

Trained artifacts live in `models/` (committed) and are loaded by the CLI at startup.
Long-term memory (what aion learns from you) lives outside the repo in `~/.aion/`.

## Corpus

The persona is trained on the Jungian books in `aion/data/books/`:

- *Man and His Symbols* — C.G. Jung (with M.-L. von Franz, J. Henderson, J. Jacobi, A. Jaffé)
- *Catafalque: Carl Jung and the End of Humanity* — Peter Kingsley
- *The Archetypes and the Collective Unconscious* — C.G. Jung
- *Modern Man in Search of a Soul* — C.G. Jung

Run `python -m aion.ingest` to rebuild `data/corpus.txt` whenever a book is added,
then retrain (see below).

## Install

```bash
python3.12 -m venv python312
source python312/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requires the `espeak-ng` system package for `/voice` (`sudo apt-get install espeak-ng`).

## Train

```bash
python train.py --epochs 120            # full training, 100+ epochs
python train.py --epochs 10 --quick     # quick smoke training
```

Each epoch trains:
1. a TF-IDF + LogisticRegression "teachings" classifier over the corpus (retrieval head),
2. a char-CNN (PyTorch) over Jung's own sentence style (generative style head).

Artifacts are written to `models/` with a training report in `models/training_report.json`.

## Run

```bash
aion
```

Then, inside the session:

| Command | Action |
|---|---|
| `/chat` | opens the chat menu → choose **talk** (conversation) or **deep** (deep analysis session) |
| `/voice` | toggles Jung's voice — spoken answers with his cadence |
| `/help` | list all commands |
| `/stats` | what aion has learned from your conversations so far |
| `/remember` | tell aion something to keep ("my name is Antony", "I live in Nairobi", "I fear spiders"...) |
| `/whoami` | what aion knows about you |
| `/forget` | wipe learned memory |
| `/quit` | leave (aion remembers your conversations) |

## Persona

aion answers as Jung himself would: first-person, measured, symbolic, pointing always
back to the inner life — the shadow, the anima/animus, complexes, dreams, individuation.
The `humanizer` module keeps the voice imperfect and alive: contractions, hedges,
rhetorical questions, the occasional German word — the way a person actually talks,
not the way a machine writes.

## How aion speaks (and remembers)

aion is built not to recite the books but to *react*:

- **Mirroring** — it picks up one of your own words and holds it up to the light:
  *"You chose the word 'drowning'; the psyche usually chooses better than we do."*
- **Introspection** — it owns an inner state: *"Something in me tightens when I hear that."*
- **Original imagery** — images composed for the moment, keyed to the topic, never
  quotations: *"A dream is a letter written in water; read it quickly or it dries into nothing."*
- **Framed remembering** — corpus lines appear rarely, and framed as memory:
  *"There is a line I have carried for years: ..."*

It also learns *you*, continuously and durably (`~/.aion/`):

- your **name** — say "my name is Antony" once (or `/remember my name is Antony`);
  every future session greets you: *"Welcome back, Antony."*
- **facts about you** — age, home, work, fears, recurring dreams — extracted from
  conversation and resurfaced when your words touch them again
- every exchange feeds the TF-IDF memory, so retrieval bends toward what you talk about

---

*"The meeting of two personalities is like the contact of two chemical substances:
if there is any reaction, both are transformed."* — C.G. Jung
