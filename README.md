# LiveScene

Open-vocabulary scene understanding from a single camera, with an event-driven
language narrator, running on a 6 GB laptop GPU.

LiveScene watches a webcam or video and keeps a live, structured account of the scene:
which objects are present (named by free text, not a fixed class list), which one is
which over time, how far away each is in metres, and what just changed. A small
vision-language model turns that into sentences and answers questions, but only when
something actually happens.

```
[   0.08s] enter    dog #2 appeared in the centre, about 3.6 m away
[   4.92s] approach dog #2 is coming closer (now 1.9 m)
narration: A small dog is running around in a public square, with people walking by
           and a person sitting on a bench.
```

## Why this design

Captioning every frame with a large model is too slow for live video and repeats
itself. LiveScene splits the work by speed:

| Layer | Runs | Model | Output |
|-------|------|-------|--------|
| Detection + segmentation | every frame | YOLOE-26s (text-prompted) | boxes, masks, labels |
| Tracking | every frame | ByteTrack | persistent object IDs |
| Depth | every 3rd frame | Depth Anything V2 Small (metric) | distance in metres |
| Scene memory | every frame | this repo, `livescene/memory.py` | object state, relations, events |
| Narration and Q&A | on events only, own thread | SmolVLM2-500M | sentences grounded in the memory |

The scene memory is the original part of this project. It converts noisy per-frame
detections into confirmed objects and discrete events (appeared, left, coming closer,
moving away, now next to), decides when the language model is worth calling, and
hands it the measured facts so it has less room to hallucinate. See
[docs/CONCEPTS.md](docs/CONCEPTS.md) for every concept used, the alternatives, and
why this set is sufficient, and [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) for a
walkthrough of every file and function.

## Measured results

Hardware: NVIDIA GTX 1660 Ti (6 GB), Intel i7-10750H, 16 GB RAM, Windows 11,
PyTorch 2.6 + CUDA 12.4, FP32. Input: 4K 25 fps video resized to 960 px wide.
Timings use `torch.cuda.synchronize()` around each stage.

| Stage | Median latency |
|-------|----------------|
| Detect + segment + track (YOLOE-26s, 640) | 56 ms |
| Depth (266 px short side, every 3rd frame) | 44 ms |
| Scene memory | 1 ms |
| 4K video decode (not needed for a webcam) | 22 ms |
| **End to end, perception only** | **92 ms median, 9.7 FPS** |
| End to end with narration running | 130 ms median, 7.3 FPS |
| Narration (40 tokens, concurrent with perception) | 4.4 s |
| Narration alone, GPU otherwise idle | 1.6 to 2.2 s |
| Peak GPU memory, all models loaded | 2.3 GB |

What these numbers say, plainly:

- The perception loop is interactive (about 10 FPS) but below 30 FPS on this card.
- Narration lags the event that triggered it by several seconds. The structured event
  line is available immediately; the sentence follows.
- Timings are sensitive to other load. With unrelated CPU-heavy jobs running on the
  same machine the same run dropped to about 5 FPS. The JSON files in `outputs/` were
  written during such a loaded run; rerun the benchmark command below on an idle
  machine to reproduce the table.
- Half precision does not help on this GPU. `tools/precision_check.py` measured the
  depth model at 29 ms in FP32 and 86 ms under FP16 autocast (outputs matched to
  0.05 %), so everything runs in FP32. On an RTX-class or Colab T4 GPU, FP16 should
  be a real speed-up; that has not been measured here.

Not measured yet: detection accuracy (mAP) on a benchmark, tracking accuracy, depth
error, and caption quality. See "Roadmap".

## Known limitations

- People in dark, close-up footage are sometimes missed or dropped by the tracker
  (visible in `outputs/demo_clip2.mp4`). YOLOE-26s is far better than YOLOE-11s here
  (person confidence 0.56 vs 0.21 on the same frame) but not reliable.
- Track IDs churn on cluttered background objects, which produces extra
  appeared / left events.
- The 500M-parameter language model sometimes states things that are not in the scene
  (for example calling a game controller a phone, following the detector's mistake).
- Metric depth comes from an indoor-trained model; outdoor distances beyond about
  10 m are rough.
- Mask outlines of multi-part objects can show stray connecting lines in the overlay.

## Setup

Requires Python 3.10+, an NVIDIA GPU, and about 8 GB of disk. Everything installs
into one folder outside this repo so it can be deleted in one step; see
[docs/INSTALLED.md](docs/INSTALLED.md).

```powershell
python -m venv $env:USERPROFILE\cv-envs\livescene\venv
$py = "$env:USERPROFILE\cv-envs\livescene\venv\Scripts\python.exe"
& $py -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install -r requirements.txt
& $py -m pip install https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
```

Model weights download automatically on first run.

## Usage

```powershell
$py = "$env:USERPROFILE\cv-envs\livescene\venv\Scripts\python.exe"

& $py run.py                                   # live webcam
& $py run.py --source video.mp4                # a video file
& $py run.py --vocab "person,coffee mug,red backpack,whiteboard"   # your own vocabulary
& $py run.py --source video.mp4 --headless --save outputs\demo.mp4 --report outputs\bench.json
& $py run.py --no-vlm                          # perception only, fastest
& $py tests\test_memory.py                     # unit tests (no GPU needed)
```

In the live window: `q` quits, `d` forces a description now, and typing a question in
the console followed by Enter asks the language model about the current view.

## Local GPU or Colab?

- **Local (this setup)** is required for the live webcam demo. Colab cannot read a
  local camera at interactive latency.
- **Colab** is the better place for benchmark evaluation (COCO / LVIS mAP, caption
  metrics) and for trying larger models (YOLOE-L, Depth Anything V2 Base, a 2B VLM),
  because a T4 or better has more memory and working FP16.

## Roadmap

1. Accuracy evaluation on Colab: open-vocabulary mAP on COCO val2017 (the earlier
   closed-set YOLOv8m project scored 0.668 mAP@0.5) and on LVIS minival.
2. Narration quality: grounded vs ungrounded prompts, scored for object hallucination.
3. TensorRT export of the detector and depth model.
4. Event precision: appearance-based re-identification to stop ID churn.
5. Pose keypoints for action events.

## Repository layout

```
run.py                   entry point (webcam, file, benchmark)
livescene/memory.py      scene memory, events, grounding prompt
livescene/perception.py  YOLOE + ByteTrack wrapper, metric depth, per-object distance
livescene/narrator.py    asynchronous vision-language model
livescene/pipeline.py    per-frame orchestration and overlay drawing
livescene/profiler.py    latency statistics
tests/test_memory.py     synthetic-track tests for the memory
tools/precision_check.py FP32 vs FP16 check for the current GPU
docs/                    concepts, code guide, install log
outputs/                 benchmark reports (annotated videos are git-ignored)
```

## Credits

Built on [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/),
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),
[SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) and
ByteTrack. Check each model's licence before commercial use (Ultralytics is AGPL-3.0).
