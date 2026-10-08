"""CHAIR: Caption Hallucination Assessment with Image Relevance (Rohrbach et al., 2018).

A caption is scanned for mentions of the 80 COCO object categories (through a synonym
list), and each mention is checked against what is really in the image.

    CHAIR_i = hallucinated object mentions / all object mentions
    CHAIR_s = captions with at least one hallucinated object / all captions
    recall  = ground-truth categories that the captions mention / all ground-truth categories

This is a dependency-free re-implementation of the official matching rules (two-word
objects, "baby elephant" is an elephant not a person, plural handling). The official
code singularises words with the `pattern` library; here a word is reduced to the first
of its candidate singular forms that the synonym list knows.
"""

import re

# Official list: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
SYNONYMS_URL = "https://raw.githubusercontent.com/LisaAnne/Hallucination/master/data/synonyms.txt"

_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light", "traffic signal", "stop light",
    "fire hydrant", "stop sign", "parking meter", "suit case", "sports ball", "baseball bat", "baseball glove",
    "tennis racket", "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear", "hair drier",
    "potted plant", "bow tie", "laptop computer", "stove top oven", "home plate", "train track",
]
_ANIMALS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"]
_VEHICLES = ["jet", "train"]


class Chair:
    def __init__(self, synonyms_path):
        self.inverse = {}  # synonym -> COCO category
        self.categories = []
        with open(synonyms_path, encoding="utf-8") as f:
            for line in f:
                words = [w.strip() for w in line.strip().split(",") if w.strip()]
                if not words:
                    continue
                self.categories.append(words[0])
                for w in words:
                    self.inverse[w] = words[0]
        self.double = {w: w for w in _DOUBLE_WORDS}
        for a in _ANIMALS:  # "baby elephant" must count as an elephant, not as a person
            self.double[f"baby {a}"] = a
            self.double[f"adult {a}"] = a
        for v in _VEHICLES:
            self.double[f"passenger {v}"] = v
        self.double["bow tie"] = "tie"
        self.double["toilet seat"] = "toilet"
        self.double["wine glas"] = "wine glass"

    @staticmethod
    def _forms(w):
        """The word itself followed by its possible singular forms."""
        out = [w]
        if w.endswith("ies") and len(w) > 4:
            out.append(w[:-3] + "y")
        if w.endswith("ves"):
            out += [w[:-3] + "fe", w[:-3] + "f"]
        if w.endswith("men"):
            out.append(w[:-3] + "man")
        if w.endswith("es"):
            out.append(w[:-2])
        if w.endswith("s") and not w.endswith("ss"):
            out.append(w[:-1])
        return out

    def objects(self, caption):
        """COCO categories mentioned in a caption, one entry per mention, in order."""
        forms = [self._forms(w) for w in re.findall(r"[a-z]+", caption.lower())]
        merged, i = [], 0
        while i < len(forms):
            pair = None
            if i + 1 < len(forms):
                pair = next((f"{a} {b}" for a in forms[i] for b in forms[i + 1] if f"{a} {b}" in self.double), None)
            if pair:
                merged.append(self.double[pair])
                i += 2
            else:
                merged.append(next((f for f in forms[i] if f in self.inverse), forms[i][0]))
                i += 1
        if "toilet" in merged and "seat" in merged:
            merged = [w for w in merged if w != "seat"]
        return [self.inverse[w] for w in merged if w in self.inverse]

    def score(self, captions, truths):
        """captions: {image_id: text}; truths: {image_id: set of COCO categories really present}."""
        mentions = halluc_mentions = halluc_caps = covered = gt_total = 0
        per_image = {}
        for iid, text in captions.items():
            objs = self.objects(text)
            truth = truths[iid]
            bad = [o for o in objs if o not in truth]
            mentions += len(objs)
            halluc_mentions += len(bad)
            halluc_caps += bool(bad)
            covered += len(set(objs) & truth)
            gt_total += len(truth)
            per_image[iid] = {"mentioned": objs, "hallucinated": sorted(set(bad))}
        n = max(1, len(captions))
        return {
            "captions": len(captions), "object_mentions": mentions, "hallucinated_mentions": halluc_mentions,
            "chair_i": halluc_mentions / max(1, mentions), "chair_s": halluc_caps / n,
            "recall": covered / max(1, gt_total),
            "avg_words": sum(len(t.split()) for t in captions.values()) / n,
        }, per_image
