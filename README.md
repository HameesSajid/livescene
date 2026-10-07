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

### Detection accuracy

COCO val2017, first 1,000 images, standard COCO box mAP (`tools/eval_coco.py`).
YOLOE is prompted with the 80 class names as text and was not trained on COCO, so its
rows are zero-shot. YOLOv8m is the supervised closed-set detector from the earlier
version of this project, run on the same images.

| Detector | Vocabulary | mAP 50-95 | mAP 50 | Small / medium / large |
|----------|------------|-----------|--------|------------------------|
| YOLOv8m (earlier project) | 80 fixed classes, trained on COCO | 0.526 | 0.694 | 0.342 / 0.590 / 0.655 |
| YOLOE-26s (default here) | any text, zero-shot | 0.383 | 0.537 | 0.224 / 0.427 / 0.521 |
| YOLOE-26m (`--weights yoloe-26m-seg.pt`) | any text, zero-shot | 0.438 | 0.605 | 0.287 / 0.477 / 0.567 |

Read this honestly: on the 80 classes COCO was built around, the supervised detector
is more accurate, by 14 points for the small open-vocabulary model and 9 for the
medium one. What the open-vocabulary detector buys is everything outside those 80
classes ("game controller", "whiteboard", "red backpack"), instance masks from the
same pass, and no retraining to change the vocabulary. The medium model recovers
5.5 mAP for roughly 15 to 25 % more detector time on this GPU (measured back to back
under the same load: 96 to 107 ms vs 121 ms per frame), so it is the better choice
when accuracy matters more than frame rate.

Not measured yet: accuracy on the full 5,000-image set and on rare categories (LVIS),
tracking accuracy, depth error, and narration quality. See "Roadmap".

## Known limitations

- People in dark, close-up footage are sometimes missed or dropped by the tracker
  (seen on a dim living-room test clip). YOLOE-26s is far better than YOLOE-11s here
  (person confidence 0.56 vs 0.21 on the same frame) but not reliable.
- Track IDs churn on cluttered background objects, which produces extra
  appeared / left events.
- The 500M-parameter language model sometimes states things that are not in the scene
  (for example calling a game controller a phone, following the detector's mistake).
- Metric depth comes from an indoor-trained model; outdoor distances beyond about
  10 m are rough.
- Mask outlines of multi-part objects can show stray connecting lines in the overlay.

## Run locally (Windows, NVIDIA GPU)

Yes, it runs fully on your own machine: no cloud service or API key is used. You need
Python 3.10 or newer, an NVIDIA GPU with a current driver, and about 9 GB of disk.
Everything (environment, model weights, datasets) stays inside the project folder,
in `.venv` and `.cache`, so deleting the folder removes all of it. See
[docs/INSTALLED.md](docs/INSTALLED.md).

**One-time setup.** Open PowerShell in the project folder:

```powershell
cd C:\Users\hamee\vscode\livescene          # or wherever you cloned it
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
```

Check that the GPU is visible (should print `True` and the GPU name):

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Run it.** Model weights (about 2.4 GB) download into `.cache` on the first run.

```powershell
.\.venv\Scripts\python.exe run.py                                  # live webcam
.\.venv\Scripts\python.exe run.py --source samples\dog_square.mp4  # a video file, in a window
.\.venv\Scripts\python.exe run.py --vocab "person,coffee mug,red backpack,whiteboard"
.\.venv\Scripts\python.exe run.py --no-vlm                         # perception only, fastest
.\.venv\Scripts\python.exe run.py --weights yoloe-26m-seg.pt       # more accurate detector
```

In the window: `q` quits, `d` forces a description now, and typing a question in the
PowerShell console followed by Enter asks the language model about the current view.

**Without a window** (save an annotated video and a latency report):

```powershell
.\.venv\Scripts\python.exe run.py --source samples\dog_square.mp4 --headless --save outputs\demo.mp4 --report outputs\bench.json
```

**Tests and evaluation:**

```powershell
.\.venv\Scripts\python.exe tests\test_memory.py            # unit tests, no GPU needed
.\.venv\Scripts\python.exe tools\eval_coco.py --n 1000     # detection mAP on a COCO subset
.\.venv\Scripts\python.exe tools\precision_check.py samples\dog_square.mp4   # FP32 vs FP16 on your GPU
```

In VS Code, open the project folder and pick `.venv` as the Python interpreter
(Ctrl+Shift+P, "Python: Select Interpreter"); then `run.py` runs with the Run button.

The `samples` folder is not part of the repository. Use any video of your own, or a
webcam.

## Run on Google Colab

Colab cannot use your webcam at interactive speed, so use it for video files, the
full accuracy evaluation, and larger models. The notebook
[notebooks/livescene_colab.ipynb](notebooks/livescene_colab.ipynb) contains every
step below as runnable cells.

1. Open [colab.research.google.com](https://colab.research.google.com), choose
   File, Upload notebook, and upload `notebooks/livescene_colab.ipynb`.
2. Runtime, Change runtime type, select a GPU (T4 or better), Save.
3. Get the code into Colab. If the repository is public:

   ```
   !git clone https://github.com/HameesSajid/livescene.git
   %cd livescene
   ```

   If it is private: on GitHub choose Code, Download ZIP, then upload the ZIP with the
   notebook's upload cell, which unpacks it.
4. Install the libraries (PyTorch with CUDA is already on Colab):

   ```
   !pip -q install -r requirements.txt
   !pip -q install https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
   ```
5. Upload a video and process it:

   ```
   from google.colab import files
   video = next(iter(files.upload()))
   !python run.py --source "$video" --headless --save outputs/colab_demo.mp4 --report outputs/colab_bench.json
   ```
6. Watch the result (re-encoded so the browser can play it):

   ```
   !ffmpeg -y -loglevel error -i outputs/colab_demo.mp4 -vcodec libx264 outputs/colab_demo_h264.mp4
   from IPython.display import Video
   Video("outputs/colab_demo_h264.mp4", embed=True, width=800)
   ```
7. Run the full COCO evaluation (all 5,000 validation images):

   ```
   !python tools/eval_coco.py --n 5000 --out outputs/coco_yoloe26s_full.json
   ```
8. Download what you want to keep with `files.download("outputs/colab_bench.json")`.
   Colab deletes the session's files when it disconnects.

## Local or Colab?

| | Local GPU | Colab |
|-|-----------|-------|
| Live webcam, questions typed live | Yes | No |
| Video files | Yes | Yes |
| Full 5,000-image COCO evaluation | Slow but possible | Recommended |
| Larger models (YOLOE-L, bigger VLM) | Limited by 6 GB | Yes |
| Half precision speed-up | No on GTX 16-series (measured) | Expected on T4 and newer |
## Roadmap

1. Full-set accuracy on Colab: `python tools/eval_coco.py --n 5000`, plus LVIS
   minival, where open-vocabulary detection is expected to show its advantage.
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
livescene/paths.py       where downloaded models and data are stored
tests/test_memory.py     synthetic-track tests for the memory
tools/precision_check.py FP32 vs FP16 check for the current GPU
tools/eval_coco.py       COCO box mAP for open-vocabulary and closed-set detectors
notebooks/               Colab notebook
docs/                    concepts, code guide, install log
outputs/                 benchmark reports (annotated videos are git-ignored)
```

## Credits

Built on [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/),
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),
[SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) and
ByteTrack. Check each model's licence before commercial use (Ultralytics is AGPL-3.0).
