"""Detection accuracy on COCO val2017 (box mAP, standard COCO protocol).

YOLOE models are prompted with the 80 COCO class names as text, so this is a
zero-shot, open-vocabulary evaluation: the detector was not trained on COCO.
Closed-set models (for example yolov8m.pt, used in the earlier project) can be
evaluated on the same images for a like-for-like comparison.

Local subset:   python tools/eval_coco.py --n 1000
Baseline:       python tools/eval_coco.py --n 1000 --weights yolov8m.pt
Full val set:   python tools/eval_coco.py --n 5000        (best run on Colab)

The first run downloads the COCO annotations (241 MB zip) and the selected images
into <cache>/data/coco. The subset is the first N image ids in sorted order, so it is
the same every time.
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from livescene.paths import CACHE  # noqa: E402

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def fetch_annotations(data_dir):
    path = os.path.join(data_dir, "instances_val2017.json")
    if not os.path.exists(path):
        print("downloading COCO annotations (241 MB) ...")
        with urllib.request.urlopen(ANN_URL) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        with z.open("annotations/instances_val2017.json") as src, open(path, "wb") as dst:
            dst.write(src.read())
    return path


def fetch_images(images, img_dir):
    def one(im):
        dst = os.path.join(img_dir, im["file_name"])
        if not os.path.exists(dst):
            urllib.request.urlretrieve(im["coco_url"], dst + ".part")
            os.replace(dst + ".part", dst)

    todo = [im for im in images if not os.path.exists(os.path.join(img_dir, im["file_name"]))]
    if todo:
        print(f"downloading {len(todo)} images ...")
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(one, todo))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="number of val2017 images (5000 = full set)")
    ap.add_argument("--weights", default="yoloe-26s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None, help="write the metrics JSON here")
    args = ap.parse_args()
    out_path = os.path.abspath(args.out) if args.out else None

    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO, YOLOE

    data_dir = os.path.join(CACHE, "data", "coco")
    img_dir = os.path.join(data_dir, "val2017")
    os.makedirs(img_dir, exist_ok=True)
    gt = COCO(fetch_annotations(data_dir))
    img_ids = sorted(gt.getImgIds())[: args.n]
    images = gt.loadImgs(img_ids)
    fetch_images(images, img_dir)

    cats = gt.loadCats(sorted(gt.getCatIds()))
    name_to_cat = {c["name"]: c["id"] for c in cats}

    os.makedirs(os.path.join(CACHE, "weights"), exist_ok=True)
    os.chdir(os.path.join(CACHE, "weights"))  # Ultralytics downloads weights into the cwd
    open_vocab = os.path.basename(args.weights).startswith("yoloe")
    if open_vocab:
        model = YOLOE(args.weights)
        names = [c["name"] for c in cats]
        model.set_classes(names, model.get_text_pe(names))
    else:
        model = YOLO(args.weights)

    dets, t_total = [], 0.0
    for i, im in enumerate(images):
        path = os.path.join(img_dir, im["file_name"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(path, imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i >= 5:  # skip warm-up
            t_total += time.perf_counter() - t0
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            dets.append({"image_id": im["id"], "category_id": name_to_cat[r.names[int(c)]],
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(images)} images")

    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    keys = ["mAP50-95", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large"]
    result = {
        "weights": os.path.basename(args.weights), "open_vocabulary_text_prompts": open_vocab,
        "images": len(images), "imgsz": args.imgsz,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ms_per_image": 1000.0 * t_total / max(1, len(images) - 5),
        **{k: float(v) for k, v in zip(keys, ev.stats[:6])},
    }
    print(json.dumps(result, indent=2))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
