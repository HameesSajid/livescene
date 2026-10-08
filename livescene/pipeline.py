"""Glue: frame in, annotated frame + events + description + answers out."""

import re
import time
from collections import deque

import cv2
import numpy as np
import torch

from .describe import propose_verify_describe, qa_prompt
from .memory import SceneMemory
from .perception import DEFAULT_VOCAB, MetricDepth, OpenVocabDetector, Tracker, observe, to_detections
from .profiler import Profiler

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _color(track_id):
    rng = np.random.default_rng(track_id * 7919 + 13)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def _label(img, text, org, color, scale=0.55):
    """Text on a dark plate, so it stays readable over any background."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _wrap(text, width_px, scale=0.6):
    rows, row = [], ""
    for word in text.split():
        if row and cv2.getTextSize(row + " " + word, FONT, scale, 1)[0][0] > width_px:
            rows.append(row)
            row = word
        else:
            row = (row + " " + word).strip()
    return rows + [row] if row else rows


class Pipeline:
    def __init__(self, frame_size, cfg, vocab=None, device=None, use_depth=True, use_vlm=True, depth_every=3,
                 fps_hint=10.0, min_gap_s=2.0, refresh_s=15.0, max_extra_vocab=30):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sync = torch.cuda.synchronize if self.device == "cuda" else None
        self.prof = Profiler()
        self.base_vocab = list(dict.fromkeys(vocab or DEFAULT_VOCAB))
        self.detector = OpenVocabDetector(cfg.detector, self.base_vocab, self.device, cfg.imgsz)
        self.tracker = Tracker(fps=fps_hint)
        self.depth_model = MetricDepth(cfg.depth, self.device, cfg.depth_side) if use_depth else None
        self.depth_every = max(1, depth_every)
        self.vlm = self.verifier = self.narrator = None
        if use_vlm:
            from .narrator import Narrator
            from .vlm import load_vlm

            self.vlm = load_vlm(cfg.vlm, max_side=cfg.vlm_side)
            # A second detector answers "is this object really there?" for the language model's
            # proposals, so those queries never disturb the vocabulary of the one that is tracking.
            self.verifier = OpenVocabDetector(cfg.verifier, ["person"], self.device, cfg.imgsz)
            self.narrator = Narrator()
        # Without a language model nothing can confirm labels, so every detection is shown.
        self.memory = SceneMemory(frame_size, require_trust=use_vlm)
        self.min_gap_s, self.refresh_s, self.max_extra_vocab = min_gap_s, refresh_s, max_extra_vocab
        self.extra_vocab = {}   # label proposed by the language model and confirmed -> last time
        self.frame_idx = 0
        self.depth = None
        self.description = ""
        self.answer = None      # dict(question, text, wall) for the most recent question
        self.description_count = 0
        self._seen = None
        self._obs = []
        self._clip = deque(maxlen=8)   # (t, frame) sampled about twice a second, for multi-frame prompts
        self._last_request_t = None
        self._asked = {}        # unconfirmed label -> last time it triggered a description
        self.log = []           # (t, kind, text) for events, descriptions and answers

    # ---------------------------------------------------------------------- loop
    def step(self, frame, t):
        """Process one BGR frame captured at time t (seconds). Returns the new events."""
        with self.prof.stage("detect+segment", self.sync):
            result = self.detector.detect(frame)
        with self.prof.stage("track"):
            self._obs = observe(result, self.tracker.update(result, frame))
        if self.depth_model is not None and self.frame_idx % self.depth_every == 0:
            with self.prof.stage("depth", self.sync):
                self.depth = self.depth_model(frame)
        with self.prof.stage("memory"):
            events = self.memory.update(t, to_detections(self._obs, self.depth, frame.shape[:2]))
        for ev in events:
            self.log.append((t, ev.kind, ev.text))
        if not self._clip or t - self._clip[-1][0] >= 0.5:
            self._clip.append((t, frame))
        if self.narrator is not None:
            self.poll(t)
            self._maybe_describe(frame, t)
        self.frame_idx += 1
        return events

    def _frames_for_vlm(self, frame):
        """The current frame, preceded by earlier ones when the profile uses several."""
        n = self.cfg.vlm_frames
        if n <= 1 or len(self._clip) < 2:
            return [frame.copy()]
        past = [f for _, f in self._clip][:-1]
        idx = np.linspace(0, len(past) - 1, min(n - 1, len(past))).round().astype(int)
        return [past[i] for i in dict.fromkeys(idx.tolist())] + [frame.copy()]

    def _maybe_describe(self, frame, t):
        if self.narrator.busy or self.narrator.question_waiting:
            return
        if self._last_request_t is None:
            reason = "start"
        else:
            since = t - self._last_request_t
            fresh = [c for c in self.memory.candidates(t) if t - self._asked.get(c, -1e9) > 30.0]
            if since >= self.min_gap_s and self.memory.pending_salience >= 1.0:
                reason = "event"
            elif since >= self.min_gap_s and fresh:
                reason = "unconfirmed object"
                for c in fresh:
                    self._asked[c] = t
            elif since >= self.refresh_s:
                reason = "refresh"
            else:
                return
        frames = self._frames_for_vlm(frame)
        context = ""
        if len(frames) > 1:
            context = f"These are {len(frames)} moments from a video in time order; the last one is now."
        # Small models copy this context into their answer word for word, so only the
        # profiles with a capable model get it, and without the internal track numbers.
        recent = [re.sub(r" #\d+", "", ev.text) for ev in self.memory.recent_events(t, 6.0)][-4:]
        if recent and not self.cfg.two_step:
            context += ("\n" if context else "") + "Changes the tracker just measured: " + "; ".join(recent) + "."
        if self.narrator.request(lambda: self._describe_job(frames, context), t):
            self._last_request_t = t
            self.memory.mark_narrated(t)

    def _describe_job(self, frames, context):
        """Runs on the language thread: propose, verify with the detector, describe."""
        out = propose_verify_describe(
            self.vlm, frames, self.verifier.check, self.cfg.verify_threshold, self.cfg.rewrite, context, self.cfg.vlm_tokens,
            two_step=self.cfg.two_step, lexicon=self.base_vocab, policy=self.cfg.policy)
        embeddings = None
        if out["accepted"] and self.verifier.weights == self.detector.weights:
            embeddings = self.verifier.embed(out["accepted"])  # slow for new words, so done here
        return {"text": out["description"], "accepted": out["accepted"], "rejected": out["rejected"],
                "embeddings": embeddings, "dropped": out["sentences_dropped"]}

    def ask(self, frame, t, question):
        """Queue a user's question about the current view. Returns False without a language model."""
        if self.narrator is None:
            return False
        frames, facts = self._frames_for_vlm(frame), self.memory.facts()
        self.answer = {"question": question, "text": None, "wall": time.time()}
        self.narrator.request(lambda: {"text": self.vlm.generate(frames, qa_prompt(question, facts), 90)},
                              t, question)
        return True

    def poll(self, t):
        """Collect a finished language job. Called every frame, and while playback is paused."""
        latest = self.narrator.latest
        if latest is None or latest is self._seen:
            return
        self._seen = latest
        if latest["question"] is not None:
            self.answer = {"question": latest["question"], "text": latest["text"], "wall": time.time()}
            self.prof.add("answer (async)", latest["latency_ms"])
            self.log.append((t, "answer", f"Q: {latest['question']}  A: {latest['text']}"))
            return
        self.description = latest["text"]
        self.description_count += 1
        self.prof.add("describe (async)", latest["latency_ms"])
        self.log.append((t, "description", self.description))
        if latest.get("rejected"):
            self.log.append((t, "rejected", "not confirmed by the detector: " + ", ".join(latest["rejected"])))
        accepted = latest.get("accepted") or []
        if accepted:
            self.memory.trust(accepted, t)
            self._update_vocab(accepted, latest.get("embeddings"), t)

    def _update_vocab(self, accepted, embeddings, t):
        for label in accepted:
            if label not in self.base_vocab:
                self.extra_vocab[label] = t
        stale = [k for k, seen in self.extra_vocab.items() if t - seen > self.memory.trust_ttl_s]
        for k in stale + sorted(self.extra_vocab, key=self.extra_vocab.get)[: -self.max_extra_vocab]:
            self.extra_vocab.pop(k, None)
        vocab = self.base_vocab + list(self.extra_vocab)
        if vocab != self.detector.vocab:
            if embeddings:
                self.detector.adopt(embeddings)
            with self.prof.stage("vocabulary change", self.sync):
                self.detector.set_vocab(vocab)

    @property
    def thinking(self):
        return self.narrator is not None and self.narrator.busy

    # ------------------------------------------------------------------ drawing
    def draw(self, frame, fps, typing=None, paused=False):
        out = frame.copy()
        overlay = out.copy()
        shown = []
        for o in self._obs:
            tr = self.memory.tracks.get(self.memory.resolve(o.track_id))
            if tr is None or not tr.confirmed:
                continue
            shown.append((o, tr))
            if tr.announced and o.mask is not None:
                overlay[o.mask.astype(bool)] = _color(tr.track_id)
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        for o, tr in shown:
            x1, y1, x2, y2 = (int(v) for v in o.box)
            if tr.announced:
                tag = tr.name + (f"  {tr.distance:.1f} m" if tr.distance is not None else "")
                cv2.rectangle(out, (x1, y1), (x2, y2), _color(tr.track_id), 2)
                _label(out, tag, (x1, max(20, y1 - 4)), (255, 255, 255))
            else:  # the detector's guess, not confirmed by the language model
                cv2.rectangle(out, (x1, y1), (x2, y2), (150, 150, 150), 1)
                _label(out, tr.label + "?", (x1, max(18, y1 - 4)), (170, 170, 170), 0.45)

        h, w = out.shape[:2]
        head = f"{fps:4.1f} FPS" + ("  paused" if paused else "") + ("  thinking..." if self.thinking else "")
        ticker = [text for _, kind, text in self.log if kind in ("enter", "exit", "approach", "recede", "near")][-4:]
        for i, line in enumerate([head] + ["* " + x for x in ticker]):
            _label(out, line, (8, 24 + 24 * i), (0, 255, 255))

        panels = []  # (rows, colour), drawn bottom-up
        if typing is not None:
            panels.append(([f"Ask: {typing}_   (Enter to send, Esc to cancel)"], (0, 255, 255)))
        if self.description:
            panels.append((_wrap(self.description, w - 20)[:4], (255, 255, 255)))
        if self.answer is not None and time.time() - self.answer["wall"] < 45:
            a = self.answer["text"] if self.answer["text"] is not None else "..."
            panels.append((_wrap("Q: " + self.answer["question"], w - 20)[:2] + _wrap("A: " + a, w - 20)[:3],
                           (120, 255, 120)))
        y = h
        for rows, colour in panels:
            top = y - 26 * len(rows) - 8
            cv2.rectangle(out, (0, top), (w, y), (0, 0, 0), -1)
            for i, r in enumerate(rows):
                cv2.putText(out, r, (10, top + 24 + 26 * i), FONT, 0.6, colour, 1, cv2.LINE_AA)
            y = top - 2
        return out

    def close(self):
        if self.narrator is not None:
            self.narrator.close()
