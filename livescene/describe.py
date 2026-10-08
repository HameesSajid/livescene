"""Propose, verify, describe.

The language model and the detector check each other, and an object is only trusted
when both report it:

1. PROPOSE  The vision-language model looks at the frame(s) and returns the objects it
            sees plus a short description, in a fixed two-field format.
2. VERIFY   Each proposed object is given to the open-vocabulary detector as a text
            prompt. Objects the detector cannot find are rejected.
3. DESCRIBE Sentences that mention a rejected object are removed (or the model is asked
            to rewrite without them), so an invented object does not reach the user.

Verified objects also become the detector's vocabulary for tracking, which removes the
old failure where a fixed word list forced a wrong nearest label (a cart called "truck").

Everything here is plain text handling, so it is unit-tested without a model.
"""

import re

PROPOSE_PROMPT = (
    "List the distinct physical objects that are clearly visible, then describe the scene.\n"
    "Use exactly this format:\n"
    "OBJECTS: comma-separated singular nouns, most prominent first, at most 12\n"
    "DESCRIPTION: two or three sentences saying what is there, where it is, and what is "
    "happening. Only mention things you can clearly see."
)

# Small models cannot hold the two-field format, so they are asked the two things separately.
DESCRIBE_PROMPT = ("Describe what is happening in this scene in two sentences. "
                   "Only mention things you can clearly see.")
LIST_PROMPT = "List the main objects visible in the image as a short comma-separated list of singular nouns."

REWRITE_PROMPT = (
    "Here is a description of the image:\n{description}\n\n"
    "These things are NOT in the image: {rejected}.\n"
    "Rewrite the description so it does not mention them. Keep everything else, add nothing new, "
    "and reply with the description only."
)

QA_PROMPT = (
    "Answer the question about the image in one or two sentences. Use only what is visible"
    "{facts_hint}. If it cannot be determined from the image, say so.\n{facts}"
    "Question: {question}"
)

_STOP = {"a", "an", "the", "some", "several", "many", "two", "three", "four", "five", "one", "of", "and"}
_NOT_OBJECTS = {"none", "nothing", "n/a", "object", "objects", "thing", "things", "scene", "image", "photo",
                "picture", "background", "foreground"}


# Different words for the same thing would otherwise become competing detector classes and
# never match the tracker's own label, so people are always called "person".
_PERSON = {"person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "kid",
           "guy", "lady", "adult", "human", "pedestrian", "player", "persons"}
_KEEP_PLURAL = {"glasses", "sunglasses", "scissors", "pants", "shorts", "jeans", "trousers", "headphones",
                "earphones", "stairs", "clothes", "goggles", "binoculars", "pliers", "tongs", "bus", "grass"}


def _singular(word):
    if word in _KEEP_PLURAL or len(word) < 4 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    return word[:-1] if word.endswith("s") else word


def clean_object(name):
    """Normalise one proposed object name; returns '' if it is not usable as a detector prompt."""
    words = [w for w in re.findall(r"[a-z][a-z\-]*", name.lower()) if w not in _STOP]
    if not words or len(words) > 3:
        return ""
    words[-1] = _singular(words[-1])
    if words[-1] in _PERSON:
        return "person"
    out = " ".join(words)
    return "" if out in _NOT_OBJECTS or len(out) < 2 else out


def parse_proposal(text, max_objects=12):
    """Split the model's reply into (objects, description). Tolerates missing or reordered
    fields: with no OBJECTS line the whole reply is treated as the description."""
    m_obj = re.search(r"objects\s*:\s*(.*?)(?=\n\s*description\s*:|$)", text, re.I | re.S)
    m_desc = re.search(r"description\s*:\s*(.*)", text, re.I | re.S)
    objects, seen = [], set()
    if m_obj:
        for part in re.split(r"[,;\n]|(?:^|\s)[-*•]\s", m_obj.group(1)):
            name = clean_object(part)
            if name and name not in seen:
                seen.add(name)
                objects.append(name)
    if m_desc:
        description = m_desc.group(1)
    elif m_obj:
        description = text[: m_obj.start()] + text[m_obj.end():]
    else:
        description = text
    description = re.sub(r"\s+", " ", re.sub(r"objects\s*:.*", "", description, flags=re.I)).strip()
    return objects[:max_objects], description


def _pattern(term):
    words = [re.escape(w) for w in term.split()]
    # allow a plural on the last word: box -> boxes, cherry -> cherries, dog -> dogs
    last = words[-1]
    stem = last[:-1] + "(?:y|ies)" if last.endswith("y") else last + "(?:s|es)?"
    return re.compile(r"\b" + r"\s+".join(words[:-1] + [stem]) + r"\b", re.I)


def mentions(text, term):
    """True if `text` names `term`, allowing a simple plural. "person" matches any word for a human."""
    if term == "person":
        return any(re.search(rf"\b{w}\b", text, re.I) for w in _PERSON)
    return bool(_pattern(term).search(text))


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def complete_sentences(text):
    """Drop a trailing fragment left when generation hit its token limit mid-sentence."""
    text = text.strip()
    if not text or text[-1] in ".!?":
        return text
    whole = split_sentences(text)[:-1]
    return " ".join(whole) if whole else text


def strip_rejected(description, rejected):
    """Drop every sentence that mentions a rejected object. Returns (text, n_dropped)."""
    kept, dropped = [], 0
    for s in split_sentences(description):
        if any(mentions(s, r) for r in rejected):
            dropped += 1
        else:
            kept.append(s)
    return " ".join(kept), dropped


def verify(objects, evidence, threshold):
    """Split proposed objects by detector evidence {name: best confidence}.

    Names that share a head noun ("table" and "dining table", "plant" and "potted plant")
    compete for the same region inside the detector, so evidence for one counts for all."""
    family = {}
    for name, conf in evidence.items():
        head = name.split()[-1]
        family[head] = max(family.get(head, 0.0), conf)
    found = [o for o in objects if family.get(o.split()[-1], 0.0) >= threshold]
    missed = [o for o in objects if o not in found]
    return found, missed


def self_check(vlm, frames, name):
    """Ask the model a direct yes/no question about one object. A model that named an object
    in passing often does not stand by it when asked directly."""
    article = "an" if name[0] in "aeiou" else "a"
    reply = vlm.generate(frames[-1:], f"Is there {article} {name} in the image? Answer yes or no.", 4)
    return reply.strip().lower().startswith("yes")

def qa_prompt(question, facts=""):
    """Prompt for a user's question; `facts` are measured values for verified objects only."""
    if facts:
        return QA_PROMPT.format(question=question, facts_hint=" and the measurements below",
                                facts="Measurements from the tracker and depth model:\n" + facts + "\n")
    return QA_PROMPT.format(question=question, facts_hint="", facts="")


def parse_list(text, max_objects=12):
    """Objects from a free-form list reply (commas, new lines or bullets)."""
    out, seen = [], set()
    for part in re.split(r"[,;\n]|(?:^|\s)[-*\u2022]\s|\band\b", re.sub(r"^[^:\n]{0,40}:", "", text.strip())):
        name = clean_object(part)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out[:max_objects]


def propose_verify_describe(vlm, frames, check, threshold=0.25, rewrite=False, context="", max_new_tokens=110,
                            two_step=False, lexicon=(), policy="either"):
    """Run the full loop on one moment.

    vlm      : object with generate(frames, prompt, max_new_tokens) -> str
    frames   : list of BGR frames, most recent last
    check    : callable(frame, [names]) -> {name: best detector confidence}
    two_step : ask for the description and the object list in separate calls (small models)
    lexicon  : known object words; any of them found in the description is verified too, so
               an object the model describes but did not list is still checked
    policy   : what happens to an object the detector cannot find.
               "detector": it is rejected. Strict, but a zero-shot detector also misses real
                           objects, and those are then cut from the description as well.
               "either"  : the model is asked about it directly and it is rejected only if
                           the model then says no. Two independent misses are needed to cut.
    Returns a dict: `description` (final text), `accepted` (found by the detector, safe to
    track), `kept` (not found, but the model stood by it), `rejected`, and `strict` (the
    text under the "detector" policy, for comparing policies on the same captions).
    """
    lead = context + "\n" if context else ""
    if two_step:
        description = re.sub(r"\s+", " ", vlm.generate(frames, lead + DESCRIBE_PROMPT, max_new_tokens)).strip()
        objects = parse_list(vlm.generate(frames, LIST_PROMPT, 40))
        raw = description
    else:
        raw = vlm.generate(frames, lead + PROPOSE_PROMPT, max_new_tokens)
        objects, description = parse_proposal(raw)
    description = complete_sentences(description)
    objects += [w for w in lexicon if w not in objects and mentions(description, w)]
    evidence = check(frames[-1], objects) if objects else {}
    accepted, missed = verify(objects, evidence, threshold)
    doubtful = [m for m in missed if mentions(description, m)]  # only these can affect the text
    strict, _ = strip_rejected(description, doubtful)
    kept = []
    if policy == "either":
        kept = [m for m in doubtful if self_check(vlm, frames, m)]
    rejected = [m for m in doubtful if m not in kept]
    final, dropped = description, 0
    if rejected:
        if rewrite:
            final = vlm.generate(frames, REWRITE_PROMPT.format(description=description, rejected=", ".join(rejected)),
                                 max_new_tokens).strip()
            # The rewrite is checked too: anything rejected that survives it is cut.
            final, dropped = strip_rejected(final, rejected)
        else:
            final, dropped = strip_rejected(description, rejected)
    fallback = ("I can see " + ", ".join(accepted) + ".") if accepted else "Nothing I can confirm in view."
    return {"description": final or fallback, "strict": strict or fallback, "raw": raw, "proposed": objects,
            "accepted": accepted, "kept": kept, "rejected": rejected,
            "evidence": {k: round(float(v), 3) for k, v in evidence.items()}, "sentences_dropped": dropped}