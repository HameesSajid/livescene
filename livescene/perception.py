"""Per-frame perception: open-vocabulary detection + segmentation + tracking (YOLOE
with ByteTrack) and monocular metric depth (Depth Anything V2)."""

import cv2
import numpy as np
import torch

from .memory import Detection

# A broad everyday vocabulary. Any noun phrase works; this is only the default prompt set.
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


class OpenVocabTracker:
    """YOLOE promptable detector/segmenter driven by text prompts, with ByteTrack IDs."""

    def __init__(self, weights="yoloe-26s-seg.pt", vocab=None, device="cuda", imgsz=640, conf=0.3,
                 tracker="bytetrack.yaml"):
        from ultralytics import YOLOE

        self.model = YOLOE(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.tracker = tracker
        self.set_vocab(vocab or DEFAULT_VOCAB)

    def set_vocab(self, vocab):
        """Re-prompt the detector. Text embeddings are computed once here and folded into
        the detection head, so the vocabulary costs nothing per frame."""
        self.vocab = list(vocab)
        self.model.set_classes(self.vocab, self.model.get_text_pe(self.vocab))

    @torch.inference_mode()
    def __call__(self, frame_bgr):
        """Returns (boxes Nx4, labels, confs, track_ids, polygons) for one frame."""
        r = self.model.track(frame_bgr, persist=True, tracker=self.tracker, imgsz=self.imgsz,
                             conf=self.conf, device=self.device, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return np.zeros((0, 4), np.float32), [], [], [], []
        boxes = r.boxes.xyxy.cpu().numpy()
        labels = [r.names[int(c)] for c in r.boxes.cls.tolist()]
        confs = r.boxes.conf.tolist()
        ids = [int(i) for i in r.boxes.id.tolist()]
        polys = list(r.masks.xy) if r.masks is not None else [None] * len(ids)
        return boxes, labels, confs, ids, polys


class MetricDepth:
    """Depth Anything V2 (metric, small). Output is a low-resolution depth map in metres."""

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


def object_distance(depth, frame_hw, box, poly):
    """Median depth over the object's mask (falls back to the centre of its box).
    The median ignores background pixels that leak in at mask edges."""
    dh, dw = depth.shape
    sx, sy = dw / frame_hw[1], dh / frame_hw[0]
    if poly is not None and len(poly) >= 3:
        m = np.zeros((dh, dw), np.uint8)
        cv2.fillPoly(m, [np.round(poly * [sx, sy]).astype(np.int32)], 1)
        vals = depth[m.astype(bool)]
        if vals.size >= 4:
            return float(np.median(vals))
    x1, y1, x2, y2 = box
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 4, (y2 - y1) / 4
    xa, xb = int(max(0, (cx - bw) * sx)), int(min(dw, (cx + bw) * sx) + 1)
    ya, yb = int(max(0, (cy - bh) * sy)), int(min(dh, (cy + bh) * sy) + 1)
    vals = depth[ya:yb, xa:xb]
    return float(np.median(vals)) if vals.size else None


def to_detections(boxes, labels, confs, ids, polys, depth, frame_hw):
    out = []
    for box, label, conf, tid, poly in zip(boxes, labels, confs, ids, polys):
        dist = object_distance(depth, frame_hw, box, poly) if depth is not None else None
        out.append(Detection(tid, label, float(conf), tuple(box.tolist()), dist))
    return out
