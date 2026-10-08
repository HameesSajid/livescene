"""Scene memory: turns per-frame tracked detections into a persistent object-level
state, a stream of discrete events, and a compact text description.

This is the piece that makes narration event-driven: the language model is only
asked to speak when the memory reports that something actually changed, and it is
given the memory's text as grounding instead of looking at pixels alone.
"""

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """One tracked detection in one frame."""

    track_id: int
    label: str
    conf: float
    box: tuple  # (x1, y1, x2, y2) in pixels
    distance: float | None = None  # metres from the camera, if depth is available


@dataclass
class Event:
    t: float
    kind: str  # enter | exit | approach | recede | near
    text: str
    track_ids: tuple
    salience: float = 1.0


@dataclass
class Track:
    track_id: int
    label: str
    first_t: float
    last_t: float
    hits: int = 0
    confirmed: bool = False  # seen often enough to be a real detection
    announced: bool = False  # confirmed and its label is trusted, so it may appear in events
    box: tuple = (0.0, 0.0, 0.0, 0.0)
    conf: float = 0.0
    distance: float | None = None
    labels: deque = field(default_factory=lambda: deque(maxlen=30))  # recent (label, confidence) votes
    dist_hist: deque = field(default_factory=lambda: deque(maxlen=90))  # (t, metres)
    motion_cooldown_until: float = 0.0

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def name(self):
        return f"{self.label} #{self.track_id}"


def _box_gap(a, b):
    """Pixel gap between two boxes (0 when they touch or overlap)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(np.hypot(dx, dy))


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / union if union > 0 else 0.0


def _plural(label, n):
    if n == 1:
        return f"1 {label}"
    if label == "person":
        return f"{n} people"
    if label.endswith(("s", "x", "ch", "sh")):
        return f"{n} {label}es"
    return f"{n} {label}s"


class SceneMemory:
    def __init__(
        self,
        frame_size,
        confirm_hits=3,
        exit_after_s=1.0,
        motion_window_s=1.5,
        motion_rel_change=0.25,
        motion_cooldown_s=3.0,
        near_gap_frac=0.03,
        near_depth_m=0.6,
        near_hold_s=0.5,
        near_release_s=1.5,
        require_trust=False,
        trust_ttl_s=90.0,
        relink_iou=0.3,
    ):
        self.w, self.h = frame_size
        self.confirm_hits = confirm_hits
        self.exit_after_s = exit_after_s
        self.motion_window_s = motion_window_s
        self.motion_rel_change = motion_rel_change
        self.motion_cooldown_s = motion_cooldown_s
        self.near_gap_px = near_gap_frac * float(np.hypot(self.w, self.h))
        self.near_depth_m = near_depth_m
        self.near_hold_s = near_hold_s
        self.near_release_s = near_release_s
        self._near_state = {}
        # With require_trust, a label only produces events once something independent of
        # the detector (the language model) has also reported it. See trust().
        self.require_trust = require_trust
        self.trust_ttl_s = trust_ttl_s
        self.relink_iou = relink_iou
        self.trusted = {}  # label -> time it was last confirmed
        self._alias = {}   # tracker id -> the id this memory shows for the same object

        self.tracks = {}
        self.near_pairs = set()
        self.events = deque(maxlen=200)
        self.pending_salience = 0.0
        self.last_narration_t = -1e9

    # ------------------------------------------------------------------ update
    def trust(self, labels, t):
        """Record that these labels were confirmed by a second source at time t."""
        for label in labels:
            self.trusted[label] = t

    def is_trusted(self, label, t):
        return not self.require_trust or t - self.trusted.get(label, -1e9) <= self.trust_ttl_s

    def resolve(self, tracker_id):
        """The id under which this memory keeps the object a tracker id refers to."""
        return self._alias.get(tracker_id, tracker_id)

    def _relink(self, d, t, present):
        """The tracker sometimes gives a still-visible object a fresh id. If a new id lands on
        a same-label track that just went missing, treat it as that track, not a new arrival."""
        best, best_iou = None, self.relink_iou
        for key, tr in self.tracks.items():
            if key in present or tr.last_t >= t or tr.label != d.label:
                continue
            iou = _iou(d.box, tr.box)
            if iou >= best_iou:
                best, best_iou = key, iou
        return best

    def update(self, t, detections):
        """Ingest one frame of detections; returns the events it produced."""
        new_events = []
        present = {self.resolve(d.track_id) for d in detections if self.resolve(d.track_id) in self.tracks}
        for d in detections:
            key = self.resolve(d.track_id)
            tr = self.tracks.get(key)
            if tr is None:
                old = self._relink(d, t, present)
                if old is not None:
                    self._alias[d.track_id] = key = old
                    present.add(old)
                    tr = self.tracks[old]
                else:
                    tr = self.tracks[key] = Track(key, d.label, first_t=t, last_t=t)
            tr.hits += 1
            tr.last_t = t
            tr.box = tuple(float(v) for v in d.box)
            tr.conf = d.conf
            # Open-vocabulary labels flicker between near-synonyms; vote over recent frames.
            tr.labels.append((d.label, d.conf))
            votes = Counter()
            for label, conf in tr.labels:
                votes[label] += conf
            tr.label = votes.most_common(1)[0][0]
            if d.distance is not None and np.isfinite(d.distance):
                # Light smoothing: per-frame monocular depth is noisy.
                tr.distance = d.distance if tr.distance is None else 0.7 * tr.distance + 0.3 * d.distance
                tr.dist_hist.append((t, tr.distance))

            if not tr.confirmed and tr.hits >= self.confirm_hits:
                tr.confirmed = True
            if tr.confirmed and not tr.announced and self.is_trusted(tr.label, t):
                tr.announced = True
                new_events.append(
                    Event(t, "enter", f"{tr.name} appeared {self._where(tr)}", (tr.track_id,), 1.0)
                )
            elif tr.announced:
                ev = self._motion_event(t, tr)
                if ev:
                    new_events.append(ev)

        for tid in [tid for tid, tr in self.tracks.items() if t - tr.last_t > self.exit_after_s]:
            tr = self.tracks.pop(tid)
            self._alias = {k: v for k, v in self._alias.items() if v != tid}
            if tr.announced:
                new_events.append(Event(t, "exit", f"{tr.name} left the view", (tid,), 1.0))

        new_events.extend(self._near_events(t))

        for ev in new_events:
            self.events.append(ev)
            self.pending_salience += ev.salience
        return new_events

    def _motion_event(self, t, tr):
        if t < tr.motion_cooldown_until or len(tr.dist_hist) < 2:
            return None
        old = next(((tt, d) for tt, d in tr.dist_hist if t - tt <= self.motion_window_s), None)
        if old is None or t - old[0] < 0.5 * self.motion_window_s or old[1] <= 0:
            return None
        rel = (tr.distance - old[1]) / old[1]
        if abs(rel) < self.motion_rel_change:
            return None
        tr.motion_cooldown_until = t + self.motion_cooldown_s
        if rel < 0:
            return Event(t, "approach", f"{tr.name} is coming closer (now {tr.distance:.1f} m)", (tr.track_id,), 1.5)
        return Event(t, "recede", f"{tr.name} is moving away (now {tr.distance:.1f} m)", (tr.track_id,), 0.75)

    def _near_events(self, t):
        live = [tr for tr in self.tracks.values() if tr.announced and tr.last_t == t]
        now = set()
        for i, a in enumerate(live):
            for b in live[i + 1 :]:
                if a.label == b.label != "person":
                    continue  # two chairs side by side is not news
                if _box_gap(a.box, b.box) > self.near_gap_px:
                    continue
                if a.distance is not None and b.distance is not None:
                    if abs(a.distance - b.distance) > self.near_depth_m:
                        continue  # overlap in the image but far apart in depth
                now.add((min(a.track_id, b.track_id), max(a.track_id, b.track_id)))
        # Hysteresis: a pair must stay adjacent for near_hold_s before it counts, and is
        # only forgotten after near_release_s apart, so jittery boxes do not re-announce.
        events = []
        for pair in now:
            a, b = self.tracks[pair[0]], self.tracks[pair[1]]
            st = self._near_state.get(pair)
            if st is None:
                # Adjacent from the moment one of them was first seen: a standing
                # relation, recorded for describe() but not announced as news.
                preexisting = min(a.hits, b.hits) <= 2 * self.confirm_hits
                st = self._near_state[pair] = {"first": t, "last": t, "held": False, "silent": preexisting}
            st["last"] = t
            if not st["held"] and t - st["first"] >= self.near_hold_s:
                st["held"] = True
                if not st["silent"]:
                    events.append(Event(t, "near", f"{a.name} is now next to {b.name}", pair, 0.75))
        stale = [p for p, st in self._near_state.items()
                 if t - st["last"] > self.near_release_s or p[0] not in self.tracks or p[1] not in self.tracks]
        for pair in stale:
            self._near_state.pop(pair)
        self.near_pairs = {p for p, st in self._near_state.items() if st["held"]}
        return events

    # ------------------------------------------------------------- description
    def _where(self, tr):
        cx, _ = tr.center
        side = "on the left" if cx < self.w / 3 else "on the right" if cx > 2 * self.w / 3 else "in the centre"
        if tr.distance is None:
            return side
        return f"{side}, about {tr.distance:.1f} m away"

    def active(self, t=None):
        trs = [tr for tr in self.tracks.values() if tr.announced]
        if t is not None:
            trs = [tr for tr in trs if t - tr.last_t <= self.exit_after_s]
        return sorted(trs, key=lambda tr: (tr.distance if tr.distance is not None else 1e9, tr.track_id))

    def describe(self, max_objects=8):
        """Zero-latency structured description of the current scene."""
        trs = self.active()
        if not trs:
            return "Nothing recognised in view."
        counts = Counter(tr.label for tr in trs)
        head = "In view: " + ", ".join(_plural(lbl, n) for lbl, n in counts.most_common()) + "."
        lines = [f"- {tr.name}: {self._where(tr)}" for tr in trs[:max_objects]]
        rels = [f"- {self.tracks[a].name} is next to {self.tracks[b].name}" for a, b in sorted(self.near_pairs)]
        return "\n".join([head] + lines + rels)

    def candidates(self, t, recent_s=1.0):
        """Labels the detector keeps reporting that nothing else has confirmed yet."""
        return sorted({tr.label for tr in self.tracks.values()
                       if tr.confirmed and not tr.announced and t - tr.last_t <= recent_s})

    def facts(self, max_objects=8):
        """Measured position and distance of trusted objects, for question answering."""
        return "\n".join(f"- {tr.label}: {self._where(tr)}" for tr in self.active()[:max_objects])

    def recent_events(self, t, horizon_s=10.0):
        return [ev for ev in self.events if t - ev.t <= horizon_s]

    # --------------------------------------------------------------- narration
    def should_narrate(self, t, min_gap_s=2.0, salience_threshold=1.0, idle_refresh_s=20.0):
        """True when enough has changed to justify a language-model call."""
        since = t - self.last_narration_t
        if since < min_gap_s:
            return False
        return self.pending_salience >= salience_threshold or (since >= idle_refresh_s and bool(self.active()))

    def mark_narrated(self, t):
        self.last_narration_t = t
        self.pending_salience = 0.0

    def grounding_prompt(self, t, question=None):
        """Text handed to the vision-language model alongside the frame."""
        parts = ["Tracked objects (from detector, tracker and depth):", self.describe()]
        evs = self.recent_events(t)
        if evs:
            parts.append("Recent events:")
            parts.extend(f"- {ev.text}" for ev in evs[-6:])
        if question:
            parts.append(f"Using the image and the facts above, answer briefly: {question}")
        else:
            parts.append(
                "Using the image and the facts above, describe what is happening in one short "
                "sentence. Do not invent objects that are not listed or visible."
            )
        return "\n".join(parts)
