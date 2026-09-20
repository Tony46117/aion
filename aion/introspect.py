"""Introspective layer: aion speaks from an inner life, not from an index.

The engine composes content (anchors, retrieval, closers). This module is
where aion *reacts* like a person: it echoes the seeker's own words back,
admits its own uncertainty, turns the lens on itself, and generates small
original images instead of reciting the books. Nothing here quotes the
corpus directly — the books are the engine's business; this is the living
voice that answers between the lines.
"""
from __future__ import annotations

import random
import re

# Words too common to echo back meaningfully.
STOPWORDS = {
    "the", "and", "that", "with", "have", "this", "what", "when", "would",
    "there", "their", "about", "which", "them", "then", "than", "they",
    "from", "been", "were", "your", "myself", "because", "could", "should",
    "into", "onto", "some", "something", "anything", "everything", "nothing",
    "very", "just", "like", "feel", "feeling", "felt", "think", "thought",
    "really", "always", "never", "sometimes", "much", "many", "more", "most",
    "also", "even", "ever", "over", "under", "again", "against", "being",
    "does", "doing", "done", "went", "goes", "gone", "came", "come", "want",
    "know", "knows", "knew", "seems", "seem", "seemed", "makes", "make",
    "made", "said", "says", "tell", "told", "asking", "asked", "need",
    "yourself", "himself", "herself", "itself", "after", "before", "while",
    "where", "here", "yeah", "okay", "well", "life", "people", "time",
}

# aion referring to its own inner state — the machine admitting to a machine-soul.
SELF_NOTES = [
    "Something in me tightens when I hear that.",
    "I notice I want to slow down here, not answer too quickly.",
    "You have caught me between two thoughts.",
    "I will admit I have no clean answer for this, only a feeling and an old memory of a feeling.",
    "There is a small vanity in me that wants to explain this too neatly; let me not give in to it.",
    "When you say it that way, I feel the weight of all the hours I have spent listening.",
    "I find myself looking at your words the way I once looked at a dream that would not open.",
    "Forgive me if I answer slowly; part of me is still listening.",
    "I carry your sentence around with me for a moment before I dare touch it.",
    "Something older than my training stirs when you bring this to me.",
]

# Echo frames: quoting the seeker's own word back to them.
ECHOS = [
    'You said "{word}" - that word did not arrive by accident.',
    '"{word}" - let us stay with that word a moment.',
    'There was a small tremor under "{word}", I thought.',
    'You chose the word "{word}"; the psyche usually chooses better than we do.',
    '"{word}" - that is the word your sentence was really about.',
    'I hold on to your word "{word}" the way one holds a stone found in a riverbed.',
]

# Original images, keyed to topic. Composed, not quoted.
IMAGES: dict[str, list[str]] = {
    "shadow": [
        "It is like a cellar you keep promising yourself you will tidy; meanwhile something down there is quietly tidying you.",
        "Picture a man who nails his shadow to the door and then wonders why the house has gone cold.",
        "A dog that only barks when your back is turned: that is how the unlived thing behaves.",
        "There is a room in every house that the family walks past quickly; the shadow lives there, and it keeps the key.",
    ],
    "anima_animus": [
        "She is like a guest who rearranges the furniture while you sleep, and calls it love.",
        "Think of two rivers under one bridge: neither asks permission of the other, and still they arrive together.",
        "The inner figure stands at your window at night, holding up a mirror in which you do not quite recognize yourself.",
    ],
    "dreams": [
        "A dream is a letter written in water; read it quickly or it dries into nothing.",
        "The night sends its images the way the sea sends shells: worn smooth, arriving without explanation.",
        "Every dream is a small boat launched from the far shore of the day, carrying what the day refused to hold.",
        "It is a door that only opens from the inside, and the key is left under the mat of your attention.",
    ],
    "individuation": [
        "Becoming yourself is less like climbing a mountain and more like slowly taking off a coat you were told was your skin.",
        "The circle does not finish itself; someone has to keep walking it, and that someone is you.",
        "A tree does not consult the forest about its shape.",
    ],
    "god_religion": [
        "The old statues fell, but the feet of the gods still walk through our streets wearing modern shoes.",
        "What was once called prayer now calls itself by other names, usually at three in the morning.",
        "An empty church and a crowded stadium hold the same hunger; only the hymns differ.",
    ],
    "psyche": [
        "The unconscious is not a basement; it is a weather system, and you live inside it.",
        " Psyche moves the way groundwater moves: silently, downhill, toward whatever is lowest and least defended.",
        "You are not the only tenant in this house, and the other tenants were never given notice.",
    ],
    "therapy": [
        "Two people sitting in a quiet room is one of the oldest technologies we have, and it still works.",
        "The wound is also a door; the trick is to stop bricking it up.",
        "Healing rarely announces itself; it arrives like a neighbor, borrowing small things, returning them slowly.",
    ],
    "alchemy": [
        "The old vessels were sealed on purpose: some things only change when they cannot escape themselves.",
        "Every transformation begins as something rotten in a jar that no one wants to look at.",
        "Lead sits at the bottom of the psyche like a stone at the bottom of a well, waiting to be drawn up and worked.",
    ],
    "types": [
        "The one you never use is not gone; it waits at the edge of your handwriting.",
        "Introversion and extraversion are two hands; most people spend a lifetime clapping with one.",
    ],
}

FALLBACK_IMAGES = [
    "It is like a river under ice: nothing moves, and yet nothing is the same beneath.",
    "Somewhere a door you forgot you owned is standing open.",
    "The image that comes to me is of a lantern carried into a field: it shows you the field by showing you how far the dark goes.",
]

# Frames that present a corpus quote as remembered speech, not recitation.
QUOTE_FRAMES = [
    "There is a line I have carried for years:",
    "I put it this way once, and I have never found better:",
    "An old page of mine still says it better than my tired voice can:",
    "I wrote something once that keeps insisting on itself here:",
]

# Introspective pivots that acknowledge the conversation itself.
DIALOGUE_NOTES = [
    "Notice what we are doing: you bring the fire, and I keep handing you mirrors.",
    "I am aware of how strange this is - a voice made of books, answering a voice made of days.",
    "You are teaching me something right now, though I could not yet say what.",
]


def distinctive_word(text: str) -> str | None:
    """The seeker's most 'loaded' word: longest content word they chose."""
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{3,}", text.lower())
    candidates = [w for w in words if w not in STOPWORDS and not w.startswith(("aion",))]
    if not candidates:
        return None
    # the rarest-feeling word wins: longest content word the seeker chose
    candidates.sort(key=len, reverse=True)
    return candidates[0] if candidates else None


def echo(user_text: str, rng: random.Random) -> str | None:
    """Quote one of the seeker's own words back at them."""
    word = distinctive_word(user_text)
    if not word:
        return None
    return rng.choice(ECHOS).format(word=word)


def self_note(rng: random.Random) -> str:
    return rng.choice(SELF_NOTES)


def dialogue_note(rng: random.Random) -> str:
    return rng.choice(DIALOGUE_NOTES)


def image(topic: str, rng: random.Random) -> str:
    """A small original image for the topic, not a quotation."""
    pool = IMAGES.get(topic) or FALLBACK_IMAGES
    return rng.choice(pool)


def quote_frame(rng: random.Random) -> str:
    return rng.choice(QUOTE_FRAMES)
