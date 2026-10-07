"""Glue: frame in, annotated frame + events + narration out."""

import cv2
import numpy as np
import torch

from .memory import SceneMemory
from .perception import MetricDepth, OpenVocabTracker, to_detections
from .profiler import Profiler


def _color(track_id):
    rng = np.random.default_rng(track_id * 7919 + 13)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def _label(img, text, org, color):
    """Text on a dark plate, so it stays readable over any background."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


class Pipeline:
    def __init__(self, frame_size, vocab=None, device=None, use_depth=True, use_vlm=True, depth_every=2,
                 det_weights="yoloe-26s-seg.pt", imgsz=640, conf=0.3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sync = torch.cuda.synchronize if self.device == "cuda" else None
        self.prof = Profiler()
        self.tracker = OpenVocabTracker(det_weights, vocab, self.device, imgsz, conf)
        self.depth_model = MetricDepth(device=self.device) if use_depth else None
        self.depth_every = max(1, depth_every)
        self.narrator = None
        if use_vlm:
            from .narrator import Narrator

            self.narrator = Narrator(device=self.device)
        self.memory = SceneMemory(frame_size)
        self.frame_idx = 0
        self.depth = None
        self.narration = ""
        self.narration_count = 0
        self._seen_narration = None
        self.log = []  # (t, kind, text) for events and narrations

    def step(self, frame, t):
        """Process one BGR frame captured at time t (seconds). Returns the new events."""
        with self.prof.stage("detect+segment+track", self.sync):
            boxes, labels, confs, ids, polys = self.tracker(frame)
        if self.depth_model is not None and self.frame_idx % self.depth_every == 0:
            with self.prof.stage("depth", self.sync):
                self.depth = self.depth_model(frame)
        with self.prof.stage("memory"):
            dets = to_detections(boxes, labels, confs, ids, polys, self.depth, frame.shape[:2])
            events = self.memory.update(t, dets)
        for ev in events:
            self.log.append((t, ev.kind, ev.text))

        if self.narrator is not None:
            if not self.narrator.busy and self.memory.should_narrate(t):
                self.narrator.request(frame, self.memory.grounding_prompt(t), t)
                self.memory.mark_narrated(t)
            self._collect_narration(t)
        self.frame_idx += 1
        self._last = (boxes, ids, polys)
        return events

    def ask(self, frame, t, question):
        if self.narrator is not None:
            self.narrator.request(frame, self.memory.grounding_prompt(t, question), t, question)

    def _collect_narration(self, t):
        latest = self.narrator.latest
        if latest is None or latest is self._seen_narration:
            return
        self._seen_narration = latest
        prefix = f"Q: {latest['question']}  A: " if latest["question"] else ""
        self.narration = prefix + latest["text"]
        self.narration_count += 1
        self.prof.add("vlm (async)", latest["latency_ms"])
        self.log.append((t, "narration", self.narration))

    # ------------------------------------------------------------------ drawing
    def draw(self, frame, fps):
        boxes, ids, polys = self._last
        out = frame.copy()
        overlay = out.copy()
        for tid, poly in zip(ids, polys):
            if poly is not None and len(poly) >= 3:
                cv2.fillPoly(overlay, [poly.astype(np.int32)], _color(tid))
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        for box, tid in zip(boxes, ids):
            tr = self.memory.tracks.get(tid)
            if tr is None or not tr.confirmed:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            tag = tr.name + (f"  {tr.distance:.1f} m" if tr.distance is not None else "")
            cv2.rectangle(out, (x1, y1), (x2, y2), _color(tid), 2)
            _label(out, tag, (x1, max(20, y1 - 4)), (255, 255, 255))

        h, w = out.shape[:2]
        lines = [f"{fps:4.1f} FPS"] + [f"* {text}" for _, kind, text in self.log[-4:] if kind != "narration"]
        for i, line in enumerate(lines):
            _label(out, line, (8, 24 + 24 * i), (0, 255, 255))
        if self.narration:
            words, rows, row = self.narration.split(), [], ""
            for word in words:
                if len(row) + len(word) > w // 11:
                    rows.append(row)
                    row = ""
                row += word + " "
            rows = (rows + [row])[-3:]
            cv2.rectangle(out, (0, h - 26 * len(rows) - 10), (w, h), (0, 0, 0), -1)
            for i, r in enumerate(rows):
                cv2.putText(out, r, (10, h - 26 * (len(rows) - i) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)
        return out

    def close(self):
        if self.narrator is not None:
            self.narrator.close()
