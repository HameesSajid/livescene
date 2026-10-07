"""Asynchronous vision-language narrator.

The VLM is far slower than the camera, so it runs on its own thread with a single
pending slot: a new request replaces any request that has not started yet, and the
perception loop never waits for language.
"""

import threading
import time

import torch
from PIL import Image


class Narrator:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", device="cuda", max_new_tokens=40,
                 image_side=384):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.image_side = image_side
        self.processor = AutoProcessor.from_pretrained(model_id)
        # By default the processor upscales and tiles the image into many crops (over a
        # thousand visual tokens). One 512 px view is 64 tokens: far faster, and the
        # fine detail is already supplied as text by the scene memory.
        ip = self.processor.image_processor
        ip.do_image_splitting = False
        ip.size = {"longest_edge": 512}
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float32).to(device).eval()

        self.latest = None  # dict(text, t_request, latency_ms, question)
        self.busy = False
        self._pending = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @torch.inference_mode()
    def generate(self, frame_bgr, prompt):
        """Blocking single call; also used directly by the benchmark."""
        img = Image.fromarray(frame_bgr[:, :, ::-1])
        img.thumbnail((self.image_side, self.image_side))
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def request(self, frame_bgr, prompt, t, question=None):
        with self._cv:
            self._pending = (frame_bgr.copy(), prompt, t, question)
            self._cv.notify()

    def _loop(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                frame, prompt, t, question = self._pending
                self._pending = None
                self.busy = True
            t0 = time.perf_counter()
            try:
                text = self.generate(frame, prompt)
            except Exception as e:  # keep the camera loop alive if generation fails
                text = f"[narrator error: {e}]"
            self.latest = {"text": text, "t_request": t, "question": question,
                           "latency_ms": (time.perf_counter() - t0) * 1000.0}
            self.busy = False

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
