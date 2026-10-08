"""Per-frame perception: open-vocabulary detection and segmentation (YOLOE) with a
vocabulary that can change while running, multi-object tracking (ByteTrack), and
monocular metric depth (Depth Anything V2)."""

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .memory import Detection

# A broad everyday vocabulary used until the language model has named what is really
# there. Labels from this list alone are treated as unconfirmed guesses.
DEFAULT_VOCAB = [
    "person", "face", "hand", "laptop", "phone", "keyboard", "computer mouse", "monitor",
    "cup", "water bottle", "book", "notebook", "pen", "backpack", "bag", "chair", "table",
    "sofa", "bed", "door", "window", "lamp", "plant", "tv", "remote control", "headphones",
    "glasses", "watch", "clock", "car", "bicycle", "motorcycle", "bus", "truck",
    "traffic light", "dog", "cat", "bird", "ball", "umbrella", "shoe", "food", "plate",
    "bowl", "fork", "knife", "spoon", "box", "paper",
]

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class Observation:
    """One tracked object in one frame."""

    track_id: int
    label: str
    conf: float
    box: np.ndarray  # (x1, y1, x2, y2) in frame pixels
    mask: np.ndarray | None  # uint8 HxW at frame resolution, or None


class OpenVocabDetector:
    """YOLOE promptable detector/segmenter driven by text prompts."""

    def __init__(self, weights="yoloe-26s-seg.pt", vocab=None, device="cuda", imgsz=640, conf=0.1):
        from ultralytics import YOLOE

        self.weights = weights
        self.model = YOLOE(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self._pe = {}  # word -> text embedding, so each word is encoded once
        self.set_vocab(vocab or DEFAULT_VOCAB)

    def embed(self, names):
        """Text embeddings for `names` as {name: tensor}. Encoding a new word takes over a
        second, so the live pipeline calls this off the camera thread."""
        missing = [n for n in dict.fromkeys(names) if n not in self._pe]
        if missing:
            pe = self.model.get_text_pe(missing)
            for i, n in enumerate(missing):
                self._pe[n] = pe[:, i:i + 1].detach()
        return {n: self._pe[n] for n in names}

    def adopt(self, embeddings):
        """Take embeddings computed by another detector that has the same weights."""
        for n, pe in embeddings.items():
            self._pe.setdefault(n, pe)

    def set_vocab(self, vocab):
        """Re-prompt the detector. The embeddings are folded into the detection head, so the
        vocabulary costs nothing per frame; with cached embeddings this takes milliseconds."""
        self.vocab = list(dict.fromkeys(vocab))
        pe = self.embed(self.vocab)
        self.model.set_classes(self.vocab, torch.cat([pe[n] for n in self.vocab], dim=1))

    @torch.inference_mode()
    def detect(self, frame_bgr, conf=None, masks=True):
        """Raw Ultralytics result for one frame (boxes, classes, confidences, masks)."""
        return self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf if conf is None else conf,
                                  device=self.device, retina_masks=masks, verbose=False)[0]

    def check(self, frame_bgr, names, conf=0.05, competitors=()):
        """Verification query: return {name: best confidence} for each of `names`.

        The detector always assigns a region to the closest prompt it was given, so asking
        about "truck" alone would let a hand cart pass as one. `competitors` are extra
        prompts that give such regions somewhere better to go. Use a detector instance that
        is not also tracking, because this replaces its vocabulary."""
        names = list(dict.fromkeys(names))
        if not names:
            return {}
        self.set_vocab(names + [c for c in competitors if c not in names])
        r = self.detect(frame_bgr, conf=conf, masks=False)
        best = {}
        if r.boxes is not None:
            for c, p in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                name = r.names[int(c)]
                best[name] = max(best.get(name, 0.0), float(p))
        return {n: best[n] for n in names if n in best}


class Tracker:
    """ByteTrack, owned by this project rather than by the Ultralytics predictor, so that
    track identities survive a change of detector vocabulary (the built-in tracker is
    discarded whenever the classes are set)."""

    def __init__(self, fps=10.0, lost_s=3.0, high=0.3, low=0.1, new=0.4):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        args = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        args.track_high_thresh = high   # confident detections drive the first association
        args.track_low_thresh = low     # weak detections may only keep an existing track alive
        args.new_track_thresh = new     # a new identity needs a clearly confident detection
        args.track_buffer = max(1, int(round(lost_s * fps)))  # frames a lost track is kept
        self.impl = BYTETracker(args)

    def update(self, result, frame_bgr):
        """Returns [(detection index, track id)] for the detections that carry a track."""
        if result.boxes is None or len(result.boxes) == 0:
            self.impl.update(_EMPTY, frame_bgr)
            return []
        rows = self.impl.update(result.boxes.cpu().numpy(), frame_bgr)
        return [(int(r[-1]), int(r[4])) for r in rows]


class _Empty:
    """Stand-in for a frame with no detections, so lost tracks still age."""

    conf = np.zeros(0, np.float32)
    cls = np.zeros(0, np.float32)
    xywh = np.zeros((0, 4), np.float32)
    xyxy = np.zeros((0, 4), np.float32)

    def __getitem__(self, idx):
        return self

    def __len__(self):
        return 0


_EMPTY = _Empty()


def observe(result, pairs):
    """Turn a detector result plus tracker assignments into Observation records."""
    if not pairs:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.tolist()
    conf = result.boxes.conf.tolist()
    masks = None
    if result.masks is not None:
        idx = torch.as_tensor([i for i, _ in pairs], device=result.masks.data.device)
        masks = (result.masks.data[idx] > 0.5).to(torch.uint8).cpu().numpy()
    return [Observation(tid, result.names[int(cls[i])], float(conf[i]), boxes[i],
                        masks[k] if masks is not None else None)
            for k, (i, tid) in enumerate(pairs)]


class MetricDepth:
    """Depth Anything V2 (metric). Output is a low-resolution depth map in metres."""

    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device="cuda",
                 short_side=266):
        from transformers import AutoModelForDepthEstimation

        self.device = device
        self.short_side = short_side  # rounded to the ViT patch size (14) below
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        self.mean, self.std = _MEAN.to(device), _STD.to(device)

    @torch.inference_mode()
    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        s = self.short_side / min(h, w)
        nh, nw = max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)
        small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small[:, :, ::-1].copy()).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        x = (x - self.mean) / self.std
        depth = self.model(pixel_values=x).predicted_depth[0]
        return depth.float().cpu().numpy()


def object_distance(depth, frame_hw, box, mask):
    """Median depth over the object's mask (falls back to the centre of its box).
    The median ignores background pixels that leak in at mask edges."""
    dh, dw = depth.shape
    if mask is not None:
        m = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
        vals = depth[m]
        if vals.size >= 4:
            return float(np.median(vals))
    sx, sy = dw / frame_hw[1], dh / frame_hw[0]
    x1, y1, x2, y2 = box
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 4, (y2 - y1) / 4
    xa, xb = int(max(0, (cx - bw) * sx)), int(min(dw, (cx + bw) * sx) + 1)
    ya, yb = int(max(0, (cy - bh) * sy)), int(min(dh, (cy + bh) * sy) + 1)
    vals = depth[ya:yb, xa:xb]
    return float(np.median(vals)) if vals.size else None


def to_detections(observations, depth, frame_hw):
    out = []
    for o in observations:
        dist = object_distance(depth, frame_hw, o.box, o.mask) if depth is not None else None
        out.append(Detection(o.track_id, o.label, o.conf, tuple(float(v) for v in o.box), dist))
    return out
