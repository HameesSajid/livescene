"""Model profiles.

The same code runs on a 6 GB laptop GPU and on a large Colab GPU; only the models change.
`auto` picks by the amount of GPU memory that is present.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    name: str
    detector: str            # YOLOE weights used for tracking
    verifier: str            # YOLOE weights used to check the language model's objects
    depth: str               # Hugging Face id of the metric depth model
    depth_side: int          # short side the depth model runs at
    vlm: str                 # spec for livescene.vlm.load_vlm
    vlm_side: int            # longest image side given to the language model
    vlm_frames: int          # frames per language-model call (more than 1 shows motion)
    vlm_tokens: int          # new tokens per description
    rewrite: bool            # ask the model to rewrite instead of only cutting sentences
    imgsz: int               # detector input size
    width: int               # frames are resized to this width
    verify_threshold: float = 0.25  # detector confidence needed to accept a proposed object
    two_step: bool = False   # ask for the description and the object list separately (small models)
    # What to do with an object the detector cannot find: "detector" cuts it, "either" first
    # asks the language model directly and cuts only if it then says no. Measured on COCO with
    # the small model, "detector" left 5 % of captions with an invented object, "either" 17 %.
    policy: str = "detector"


PROFILES = {
    # Fits a 6 GB GPU and keeps the camera loop interactive.
    "laptop": Config(
        name="laptop", detector="yoloe-26s-seg.pt", verifier="yoloe-26s-seg.pt",
        depth="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", depth_side=266,
        vlm="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", vlm_side=512, vlm_frames=1, vlm_tokens=60,
        rewrite=False, imgsz=640, width=960, two_step=True),
    # About 12-16 GB (Colab T4): a 3B language model in half precision, large detector.
    "colab-t4": Config(
        name="colab-t4", detector="yoloe-26l-seg.pt", verifier="yoloe-26l-seg.pt",
        depth="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", depth_side=392,
        vlm="Qwen/Qwen2.5-VL-3B-Instruct@fp16", vlm_side=768, vlm_frames=2, vlm_tokens=160,
        rewrite=True, imgsz=960, width=1280),
    # 22 GB or more (Colab L4 / A100): a 7B language model, the largest detector and depth model.
    "colab": Config(
        name="colab", detector="yoloe-26x-seg.pt", verifier="yoloe-26x-seg.pt",
        depth="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", depth_side=518,
        vlm="Qwen/Qwen2.5-VL-7B-Instruct@fp16", vlm_side=896, vlm_frames=3, vlm_tokens=180,
        rewrite=True, imgsz=960, width=1280),
}


def pick(name="auto", **overrides):
    """Resolve a profile name (or `auto`) and apply any explicit overrides that are not None."""
    if name == "auto":
        import torch

        gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
        name = "colab" if gb >= 22 else "colab-t4" if gb >= 12 else "laptop"
    cfg = PROFILES[name]
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
