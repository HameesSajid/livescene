"""Synthetic-track tests for SceneMemory. Run: python tests/test_memory.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from livescene import Detection, SceneMemory  # noqa: E402

FPS = 30.0


def run(mem, frames):
    """frames: list of per-frame detection lists. Returns all events."""
    out = []
    for i, dets in enumerate(frames):
        out.extend(mem.update(i / FPS, dets))
    return out


def test_enter_needs_confirmation_and_exit_fires_once():
    mem = SceneMemory((1280, 720))
    person = Detection(1, "person", 0.9, (100, 100, 300, 600), 2.0)
    events = run(mem, [[person]] * 2)
    assert events == [], "a 2-frame flicker must not be announced"
    events = run(mem, [[person]] * 10 + [[]] * 60)
    kinds = [e.kind for e in events]
    assert kinds == ["enter", "exit"], kinds
    assert "on the left" in events[0].text and "2.0 m" in events[0].text


def test_label_flicker_is_voted_out():
    mem = SceneMemory((1280, 720))
    frames = [[Detection(7, "cup" if i % 5 else "mug", 0.8, (600, 300, 700, 400), 1.0)] for i in range(30)]
    run(mem, frames)
    assert mem.tracks[7].label == "cup"
    assert mem.describe().startswith("In view: 1 cup.")


def test_approach_event_with_cooldown():
    mem = SceneMemory((1280, 720))
    frames = [[Detection(2, "person", 0.9, (500, 100, 800, 700), 4.0 - 2.5 * min(i, 60) / 60)] for i in range(120)]
    events = run(mem, frames)
    approaches = [e for e in events if e.kind == "approach"]
    assert 1 <= len(approaches) <= 2, [e.text for e in events]
    assert not [e for e in events if e.kind == "recede"]


def test_near_respects_depth():
    def frames(cup_dist):
        out = []
        for i in range(90):
            x = 900 - min(i, 60) * 8  # cup slides toward the person
            out.append(
                [
                    Detection(1, "person", 0.9, (300, 100, 500, 700), 2.0),
                    Detection(2, "cup", 0.8, (x, 400, x + 60, 470), cup_dist),
                ]
            )
        return out

    same_depth = [e for e in run(SceneMemory((1280, 720)), frames(2.1)) if e.kind == "near"]
    far_behind = [e for e in run(SceneMemory((1280, 720)), frames(5.0)) if e.kind == "near"]
    assert len(same_depth) == 1, same_depth
    assert far_behind == []


def test_narration_trigger_is_event_driven():
    mem = SceneMemory((1280, 720))
    person = Detection(1, "person", 0.9, (100, 100, 300, 600), 2.0)
    calls = 0
    for i in range(int(15 * FPS)):  # 15 s of a static scene
        t = i / FPS
        mem.update(t, [person])
        if mem.should_narrate(t):
            calls += 1
            mem.mark_narrated(t)
    assert calls == 1, f"static scene should be narrated once, got {calls}"
    assert "person #1" in mem.grounding_prompt(15.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
