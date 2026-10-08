"""Tests for the propose / verify / describe text logic. Run: python tests/test_describe.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from livescene.describe import (  # noqa: E402
    clean_object, complete_sentences, mentions, parse_proposal, propose_verify_describe, qa_prompt, strip_rejected, verify)


def test_parse_well_formed_reply():
    objs, desc = parse_proposal("OBJECTS: a dog, Three People, bench, the trash can\n"
                                "DESCRIPTION: A dog runs across a square. People sit on a bench.")
    assert objs == ["dog", "person", "bench", "trash can"]
    assert desc == "A dog runs across a square. People sit on a bench."


def test_parse_tolerates_messy_replies():
    objs, desc = parse_proposal("Objects:\n- dog\n- wooden bench\n\nDescription: A dog near a bench.")
    assert objs == ["dog", "wooden bench"] and desc == "A dog near a bench."
    objs, desc = parse_proposal("A dog near a bench.")  # no fields at all
    assert objs == [] and desc == "A dog near a bench."
    objs, _ = parse_proposal("OBJECTS: none\nDESCRIPTION: An empty room.")
    assert objs == []


def test_clean_object_rejects_junk():
    assert clean_object("  The Red Backpack ") == "red backpack"
    assert clean_object("a very long phrase that is not an object") == ""
    assert clean_object("background") == "" and clean_object("12") == ""


def test_clean_object_canonicalises():
    # Every word for a human becomes "person", so it matches the tracker's own label.
    assert [clean_object(x) for x in ("two women", "a young man", "children", "People")] == ["person"] * 4
    assert clean_object("wine glasses") == "wine glasses" and clean_object("benches") == "bench"
    assert clean_object("cherries") == "cherry" and clean_object("dogs") == "dog" and clean_object("bus") == "bus"


def test_mentions_handles_plurals_and_word_boundaries():
    assert mentions("Two trucks are parked.", "truck")
    assert mentions("Boxes of cherries", "box") and mentions("Boxes of cherries", "cherry")
    assert mentions("a trash  can stands there", "trash can")
    assert not mentions("A cartoon on the wall", "cart")
    assert not mentions("The structure is tall", "truck")
    assert mentions("Two women are talking.", "person") and not mentions("A personal item", "person")


def test_strip_rejected_removes_only_offending_sentences():
    text = "A dog runs across the square. A truck is passing by. People sit on a bench."
    out, n = strip_rejected(text, ["truck"])
    assert out == "A dog runs across the square. People sit on a bench." and n == 1


def test_complete_sentences_drops_a_cut_off_tail():
    assert complete_sentences("Two people sit. They are looking at") == "Two people sit."
    assert complete_sentences("Two people sit.") == "Two people sit."
    assert complete_sentences("a single unfinished fragment") == "a single unfinished fragment"


def test_verify_splits_by_threshold():
    acc, rej = verify(["dog", "truck", "bench"], {"dog": 0.7, "truck": 0.1}, 0.25)
    assert acc == ["dog"] and rej == ["truck", "bench"]


def test_verify_shares_evidence_within_a_noun_family():
    # The detector gave the region to "dining table"; plain "table" must not be rejected for that.
    acc, rej = verify(["table", "dining table", "plant"], {"dining table": 0.6, "potted plant": 0.1}, 0.25)
    assert acc == ["table", "dining table"] and rej == ["plant"]


class FakeVLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, frames, prompt, max_new_tokens):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_loop_drops_an_invented_object():
    vlm = FakeVLM(["OBJECTS: dog, truck, bench\nDESCRIPTION: A dog runs. A truck passes by. People sit on a bench."])
    out = propose_verify_describe(vlm, ["frame"], lambda frame, names: {"dog": 0.8, "bench": 0.5, "truck": 0.05},
                                  policy="detector")
    assert out["accepted"] == ["dog", "bench"] and out["rejected"] == ["truck"]
    assert out["description"] == "A dog runs. People sit on a bench." and out["sentences_dropped"] == 1


def test_loop_asks_the_model_before_cutting():
    # The detector misses both the truck and the bear; the model stands by the bear only.
    vlm = FakeVLM(["OBJECTS: bear, truck\nDESCRIPTION: A bear sits on grass. A truck passes by.", "Yes.", "No."])
    out = propose_verify_describe(vlm, ["frame"], lambda f, n: {}, policy="either")
    assert out["kept"] == ["bear"] and out["rejected"] == ["truck"] and out["accepted"] == []
    assert out["description"] == "A bear sits on grass."
    assert out["strict"] == "Nothing I can confirm in view."
    assert "bear" in vlm.prompts[1] and "truck" in vlm.prompts[2]


def test_loop_rewrite_is_checked_again():
    vlm = FakeVLM(["OBJECTS: dog, truck\nDESCRIPTION: A dog runs beside a truck.",
                   "A dog runs. The truck is gone."])  # a rewrite that still names the truck
    out = propose_verify_describe(vlm, ["frame"], lambda f, n: {"dog": 0.9}, rewrite=True, policy="detector")
    assert out["description"] == "A dog runs."
    assert "truck" in vlm.prompts[1]


def test_loop_falls_back_when_everything_is_rejected():
    vlm = FakeVLM(["OBJECTS: unicorn\nDESCRIPTION: A unicorn stands here."])
    out = propose_verify_describe(vlm, ["frame"], lambda f, n: {}, policy="detector")
    assert out["description"] == "Nothing I can confirm in view."


def test_qa_prompt_includes_facts_only_when_given():
    assert "Measurements" not in qa_prompt("How many people?")
    p = qa_prompt("How far is the dog?", "- dog: centre, 2.0 m away")
    assert "2.0 m" in p and p.rstrip().endswith("Question: How far is the dog?")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
