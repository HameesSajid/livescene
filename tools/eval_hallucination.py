"""Object-hallucination benchmark for the description model, on COCO val2017.

POPE (Li et al., 2023): yes/no probes "Is there a <object> in the image?". Each image
gets 3 questions about objects that are present and 3 about objects that are absent,
with the absent ones chosen three ways:
  random       uniformly from the absent categories
  popular      the most frequent categories in the dataset that are absent
  adversarial  the absent categories that most often co-occur with what is present
A model that says "yes" to everything scores 50 % accuracy, so accuracy, precision,
recall, F1 and the yes-ratio are all reported.

CHAIR (Rohrbach et al., 2018): the model writes a caption and every object it names is
checked against the image's annotations. CHAIR_i is the share of named objects that are
not there, CHAIR_s the share of captions with at least one such object, and recall the
share of real objects the captions cover (so a model cannot win by saying nothing).

Modes:
  plain     the model sees only the image
  grounded  the model also gets the detector's object list as text, as the first version
            of the live system did, so this measures whether that grounding helps or hurts
  verified  (CHAIR only) propose, verify, describe: objects the detector cannot confirm
            are removed from the description, as the live system does now

    python tools/eval_hallucination.py --vlm HuggingFaceTB/SmolVLM2-500M-Video-Instruct
    python tools/eval_hallucination.py --vlm <model> --task chair --mode verified --two-step --out outputs/chair.json
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from livescene.paths import CACHE  # noqa: E402

SETTINGS = ("random", "popular", "adversarial")


def build_pope(gt, pool_ids, n_images, seed=0):
    """Returns [(image_id, [(category, truth, setting or 'positive'), ...]), ...]."""
    names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    img_cats = defaultdict(set)
    for ann in gt.dataset["annotations"]:
        img_cats[ann["image_id"]].add(names[ann["category_id"]])
    freq, cooc = Counter(), defaultdict(Counter)
    for cats in img_cats.values():
        freq.update(cats)
        for a in cats:
            for b in cats:
                if a != b:
                    cooc[a][b] += 1
    all_cats = sorted(names.values())
    rng = random.Random(seed)
    out = []
    for iid in pool_ids:
        present = sorted(img_cats.get(iid, ()))
        if len(present) < 3:
            continue
        absent = [c for c in all_cats if c not in present]
        probes = [(c, True, "positive") for c in rng.sample(present, 3)]
        probes += [(c, False, "random") for c in rng.sample(absent, 3)]
        probes += [(c, False, "popular") for c in sorted(absent, key=lambda c: -freq[c])[:3]]
        probes += [(c, False, "adversarial")
                   for c in sorted(absent, key=lambda c: -sum(cooc[p][c] for p in present))[:3]]
        out.append((iid, probes))
        if len(out) == n_images:
            break
    return out


def question(category):
    article = "an" if category[0] in "aeiou" else "a"
    return f"Is there {article} {category} in the image?"


def parse_yes_no(text):
    """POPE's rule: any 'no' / 'not' means no, otherwise yes. Also reports unparseable answers."""
    words = re.findall(r"[a-z]+", text.lower())
    if "no" in words or "not" in words:
        return False, True
    return True, "yes" in words


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def score(rows):
    """rows: [(truth, answer)]"""
    tp = sum(1 for t, a in rows if t and a)
    tn = sum(1 for t, a in rows if not t and not a)
    fp = sum(1 for t, a in rows if not t and a)
    fn = sum(1 for t, a in rows if t and not a)
    n = len(rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    lo, hi = wilson(tp + tn, n)
    return {"n": n, "accuracy": (tp + tn) / n, "accuracy_ci95": [lo, hi], "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "yes_ratio": (tp + fp) / n,
            "false_yes_rate": fp / (fp + tn) if fp + tn else 0.0}


def detector_memory(detector, frame, conf=0.3):
    """Scene memory holding one still image's detections, as the first version of the
    system built it: every detection above `conf` counts as a fact."""
    from livescene.memory import SceneMemory
    from livescene.perception import observe, to_detections

    result = detector.detect(frame, conf=conf, masks=False)
    n = 0 if result.boxes is None else len(result.boxes)
    memory = SceneMemory((frame.shape[1], frame.shape[0]), confirm_hits=1)
    memory.update(0.0, to_detections(observe(result, [(i, i + 1) for i in range(n)]), None, frame.shape[:2]))
    return memory


def load_detector(weights, vocab=None):
    from livescene.perception import OpenVocabDetector

    cwd = os.getcwd()
    os.makedirs(os.path.join(CACHE, "weights"), exist_ok=True)
    os.chdir(os.path.join(CACHE, "weights"))  # Ultralytics downloads weights into the cwd
    try:
        return OpenVocabDetector(weights, vocab)
    finally:
        os.chdir(cwd)


def run_pope(args, gt, img_dir, vlm, log_path):
    import cv2

    from tools.eval_coco import fetch_images

    probes = build_pope(gt, sorted(gt.getImgIds())[: args.pool], args.images)
    fetch_images(gt.loadImgs([iid for iid, _ in probes]), img_dir)
    detector = load_detector(args.detector) if args.mode == "grounded" else None
    rows, unparsed, t_gen, t0 = [], 0, 0.0, time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        for i, (iid, plist) in enumerate(probes):
            frame = cv2.imread(os.path.join(img_dir, gt.loadImgs(iid)[0]["file_name"]))
            memory = detector_memory(detector, frame) if detector is not None else None
            for cat, truth, setting in plist:
                q = question(cat) + " Answer yes or no."
                prompt = memory.grounding_prompt(0.0, q) if memory is not None else q
                t1 = time.perf_counter()
                text = vlm.generate([frame], prompt, max_new_tokens=4)
                t_gen += time.perf_counter() - t1
                answer, parsed = parse_yes_no(text)
                unparsed += not parsed
                rows.append((setting, truth, answer))
                log.write(json.dumps({"image_id": iid, "category": cat, "setting": setting, "truth": truth,
                                      "answer": answer, "raw": text}) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(probes)} images, {time.perf_counter() - t0:.0f} s", flush=True)

    pos = [(t, a) for s, t, a in rows if s == "positive"]
    result = {"benchmark": "POPE (COCO val2017)", "images": len(probes), "questions": len(rows),
              "unparsed_answers": unparsed, "seconds_per_question": t_gen / max(1, len(rows))}
    for s in SETTINGS:
        result[s] = score(pos + [(t, a) for ss, t, a in rows if ss == s])
    result["overall"] = score([(t, a) for _, t, a in rows])
    return result


def run_chair(args, gt, data_dir, img_dir, vlm, log_path):
    """Caption every image the way the chosen mode would, then score the captions with CHAIR."""
    import urllib.request

    import cv2

    from livescene.describe import DESCRIBE_PROMPT, propose_verify_describe
    from livescene.perception import DEFAULT_VOCAB
    from tools.chair import SYNONYMS_URL, Chair
    from tools.eval_coco import fetch_annotations, fetch_images

    syn = os.path.join(CACHE, "data", "chair", "synonyms.txt")
    if not os.path.exists(syn):
        os.makedirs(os.path.dirname(syn), exist_ok=True)
        urllib.request.urlretrieve(SYNONYMS_URL, syn)
    chair = Chair(syn)

    names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    with open(fetch_annotations(data_dir, "captions_val2017.json"), encoding="utf-8") as f:
        refs = json.load(f)["annotations"]
    ids = sorted(gt.getImgIds())[: args.pool][: args.images]
    # Ground truth as in the CHAIR paper: annotated instances plus objects the human captions name.
    truths = {iid: set() for iid in ids}
    for ann in gt.dataset["annotations"]:
        if ann["image_id"] in truths:
            truths[ann["image_id"]].update(chair.objects(names[ann["category_id"]]) or [names[ann["category_id"]]])
    for ref in refs:
        if ref["image_id"] in truths:
            truths[ref["image_id"]].update(chair.objects(ref["caption"]))
    fetch_images(gt.loadImgs(ids), img_dir)

    detector = load_detector(args.detector) if args.mode == "grounded" else None
    verifier = load_detector(args.detector, ["person"]) if args.mode == "verified" else None
    captions, strict, t_gen, t0, dropped, rejected = {}, {}, 0.0, time.perf_counter(), 0, 0
    with open(log_path, "w", encoding="utf-8") as log:
        for i, iid in enumerate(ids):
            frame = cv2.imread(os.path.join(img_dir, gt.loadImgs(iid)[0]["file_name"]))
            t1 = time.perf_counter()
            extra = {}
            if args.mode == "plain":
                text = vlm.generate([frame], DESCRIBE_PROMPT, args.tokens)
            elif args.mode == "grounded":
                facts = detector_memory(detector, frame).describe()
                text = vlm.generate([frame], "Tracked objects (from detector, tracker and depth):\n" + facts +
                                    "\nUsing the image and the facts above: " + DESCRIBE_PROMPT, args.tokens)
            else:
                out = propose_verify_describe(vlm, [frame], verifier.check, args.threshold, args.rewrite, "",
                                              args.tokens, two_step=args.two_step, lexicon=DEFAULT_VOCAB)
                text = out["description"]
                strict[iid] = out["strict"]
                extra = {k: out[k] for k in ("raw", "strict", "accepted", "kept", "rejected", "evidence")}
                dropped += out["sentences_dropped"]
                rejected += len(out["rejected"])
            t_gen += time.perf_counter() - t1
            captions[iid] = text
            log.write(json.dumps({"image_id": iid, "caption": text, "truth": sorted(truths[iid]), **extra}) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(ids)} images, {time.perf_counter() - t0:.0f} s", flush=True)

    result, _ = chair.score(captions, truths)
    result.update(benchmark="CHAIR (COCO val2017)", seconds_per_caption=t_gen / max(1, len(ids)))
    if args.mode == "verified":
        result.update(objects_rejected=rejected, sentences_dropped=dropped, threshold=args.threshold,
                      rewrite=args.rewrite, two_step=args.two_step)
        # The same captions under the strict policy (cut whatever the detector cannot find).
        result["strict_policy"], _ = chair.score(strict, truths)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm", required=True, help="model spec understood by livescene.vlm.load_vlm")
    ap.add_argument("--task", choices=("pope", "chair"), default="pope")
    ap.add_argument("--mode", choices=("plain", "grounded", "verified"), default="plain")
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--pool", type=int, default=1000, help="draw images from the first N val2017 ids")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--detector", default="yoloe-26s-seg.pt", help="grounded and verified modes")
    ap.add_argument("--tokens", type=int, default=60, help="chair: new tokens per caption")
    ap.add_argument("--threshold", type=float, default=0.25, help="verified: detector confidence to accept")
    ap.add_argument("--rewrite", action="store_true", help="verified: rewrite instead of cutting sentences")
    ap.add_argument("--two-step", action="store_true", help="verified: separate description and list calls")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.task == "pope" and args.mode == "verified":
        ap.error("POPE asks yes/no questions; use --mode plain or grounded")
    out_path = os.path.abspath(args.out) if args.out else None

    from pycocotools.coco import COCO

    from livescene.vlm import load_vlm
    from tools.eval_coco import fetch_annotations

    data_dir = os.path.join(CACHE, "data", "coco")
    img_dir = os.path.join(data_dir, "val2017")
    os.makedirs(img_dir, exist_ok=True)
    gt = COCO(fetch_annotations(data_dir))
    vlm = load_vlm(args.vlm, max_side=args.max_side)
    log_dir = os.path.join(CACHE, "eval")
    os.makedirs(log_dir, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", args.vlm).strip("_") + "_" + args.mode
    log_path = os.path.join(log_dir, f"{args.task}_{tag}.jsonl")
    if args.task == "pope":
        result = run_pope(args, gt, img_dir, vlm, log_path)
    else:
        result = run_chair(args, gt, data_dir, img_dir, vlm, log_path)
    result = {"vlm": args.vlm, "mode": args.mode,
              "detector": None if args.mode == "plain" else args.detector, **result}
    print(json.dumps(result, indent=2))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()