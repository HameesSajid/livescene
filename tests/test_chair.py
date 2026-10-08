"""Tests for the CHAIR scorer. Run: python tests/test_chair.py

Uses a small inline synonym list, so no download is needed.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools.chair import Chair  # noqa: E402

SYNONYMS = """person, man, woman, people, child, baby, girl, boy
dog, puppy
elephant
truck, pickup
wine glass
cup, mug
knife
tie
toilet
cell phone, phone, smartphone
bench
"""


def make():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(SYNONYMS)
    try:
        return Chair(f.name)
    finally:
        os.unlink(f.name)


def test_synonyms_and_plurals_map_to_categories():
    c = make()
    assert c.objects("Two women and a puppy sit on benches.") == ["person", "dog", "bench"]
    assert c.objects("Knives and mugs on a table") == ["knife", "cup"]


def test_two_word_objects():
    c = make()
    assert c.objects("She holds two wine glasses and a cell phone.") == ["wine glass", "cell phone"]
    assert c.objects("A man in a bow tie") == ["person", "tie"]
    assert c.objects("a toilet seat is up") == ["toilet"]


def test_baby_animal_is_not_a_person():
    c = make()
    assert c.objects("A baby elephant walks.") == ["elephant"]
    assert c.objects("A baby sleeps.") == ["person"]


def test_scores():
    c = make()
    captions = {1: "A man walks a dog past a truck.", 2: "A woman holds a cup.", 3: "An empty street."}
    truths = {1: {"person", "dog", "bench"}, 2: {"person", "cup"}, 3: {"truck"}}
    s, per_image = c.score(captions, truths)
    assert s["object_mentions"] == 5 and s["hallucinated_mentions"] == 1
    assert abs(s["chair_i"] - 1 / 5) < 1e-9
    assert abs(s["chair_s"] - 1 / 3) < 1e-9
    assert abs(s["recall"] - 4 / 6) < 1e-9  # person, dog, person, cup of 6 ground-truth entries
    assert per_image[1]["hallucinated"] == ["truck"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
