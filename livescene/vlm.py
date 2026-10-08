"""Vision-language model backends.

Every backend exposes the same blocking call,

    generate(frames_bgr, prompt, max_new_tokens) -> str

so the narrator and the evaluation tools can swap models without caring how they run.
`frames_bgr` is a list of BGR numpy frames (one for a still, several for a short clip).
"""

import base64
import json
import urllib.request

import cv2
import torch
from PIL import Image


def _to_pil(frame_bgr, max_side):
    img = Image.fromarray(frame_bgr[:, :, ::-1])
    img.thumbnail((max_side, max_side))
    return img


class HFVLM:
    """Any Hugging Face image-text-to-text chat model, run in-process with PyTorch."""

    def __init__(self, model_id, device="cuda", dtype="float32", max_side=512, load_4bit=False):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = model_id
        self.device = device
        self.max_side = max_side
        self.processor = AutoProcessor.from_pretrained(model_id)
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None and hasattr(ip, "do_image_splitting"):
            # SmolVLM-style processors upscale and tile the image into many crops (over a
            # thousand visual tokens). One view is far faster.
            ip.do_image_splitting = False
            ip.size = {"longest_edge": max_side}
        kwargs = {"dtype": getattr(torch, dtype)}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype))
            kwargs["device_map"] = {"": device}
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        if not load_4bit:
            self.model = self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, frames_bgr, prompt, max_new_tokens=48):
        content = [{"type": "image", "image": _to_pil(f, self.max_side)} for f in frames_bgr]
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
            return_dict=True, processor_kwargs={"return_tensors": "pt"}).to(self.device)
        dtype = next(self.model.parameters()).dtype
        inputs = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class ServerVLM:
    """A model behind an OpenAI-compatible chat endpoint (for example llama.cpp's server)."""

    def __init__(self, url, model="default", max_side=512, timeout=120):
        self.name = f"{url} ({model})"
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.max_side = max_side
        self.timeout = timeout

    def generate(self, frames_bgr, prompt, max_new_tokens=48):
        content = []
        for f in frames_bgr:
            h, w = f.shape[:2]
            s = min(1.0, self.max_side / max(h, w))
            small = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else f
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}})
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": max_new_tokens, "temperature": 0}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()


def load_vlm(spec, **kwargs):
    """`spec` is a Hugging Face model id, or `server:<base url>` for an HTTP endpoint.
    Append `@4bit` or `@fp16` to a model id to change how it is loaded."""
    if spec.startswith("server:"):
        return ServerVLM(spec[len("server:"):], **kwargs)
    model_id, _, mode = spec.partition("@")
    if mode == "4bit":
        return HFVLM(model_id, load_4bit=True, **kwargs)
    if mode == "fp16":
        return HFVLM(model_id, dtype="float16", **kwargs)
    return HFVLM(model_id, **kwargs)
