# LiveScene

Scene description from a single camera that checks itself: a vision-language model says
what it sees, an open-vocabulary detector has to confirm each object, and only what both
agree on is described, tracked and measured.

LiveScene watches a webcam or video and keeps a live account of the scene: which objects
are present (named in free text, not from a fixed class list), which one is which over
time, how far away each is in metres, what just changed, and a short description in
plain language. You can type a question about the scene and get an answer.

```
[   1.92s] description Two people are sitting on a couch in a living room, playing video games.
[   3.56s] answer      Q: What are the people doing?  A: the people are playing a game on a device
[   4.92s] approach    dog #2 is coming closer (now 1.9 m)
[   5.40s] rejected    not confirmed by the detector: remote
```

## How it works

Two loops run at different speeds.

| Loop | Runs | What it does |
|------|------|--------------|
| Fast | every frame | Detect and segment (YOLOE, text-prompted), track (ByteTrack), estimate metric depth (Depth Anything V2), update the scene memory, draw |
| Slow | when something changed, on its own thread | Propose, verify, describe (below), and answer questions |

**Propose, verify, describe.** This is the part that controls hallucination.

1. *Propose.* The language model looks at the frame and returns the objects it sees and
   a short description.
2. *Verify.* Each object is given to a second detector as a text prompt. If the detector
   cannot find it, it is rejected.
3. *Describe.* Sentences that mention a rejected object are removed (or the model
   rewrites without them).
4. *Feed back.* Confirmed objects become the tracking detector's vocabulary, so the
   tracker follows what is actually in this scene instead of a fixed word list.

A label produces events ("appeared", "left", "coming closer", "next to") only once both
models have reported it. Detections that only the detector believes in are drawn as a
thin grey box with a question mark and are otherwise ignored.

The first version of this project did the opposite: it handed the detector's labels to
the language model as facts. The benchmark below shows that this made hallucination
worse, which is why the direction of trust was reversed. See
[docs/CONCEPTS.md](docs/CONCEPTS.md) for the reasoning and
[docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) for every file and function.

## Profiles

The same code runs with small or large models. `--profile auto` (the default) picks by
GPU memory.

| Profile | GPU memory | Detector | Depth | Language model |
|---------|------------|----------|-------|----------------|
| `laptop` | 6 GB | YOLOE-26s | Depth Anything V2 Small | SmolVLM2-500M |
| `colab-t4` | 12 to 16 GB | YOLOE-26l | Depth Anything V2 Base | Qwen2.5-VL-3B, half precision |
| `colab` | 22 GB or more | YOLOE-26x | Depth Anything V2 Large | Qwen2.5-VL-7B, half precision, 3 frames per call |

Only the `laptop` profile has been run and measured so far. The two Colab profiles use
the same code path with larger checkpoints of the same model families; their numbers
come from running the notebook on Colab.

## Measured results

All numbers below are for the `laptop` profile on an NVIDIA GTX 1660 Ti (6 GB),
Intel i7-10750H, 16 GB RAM, Windows 11, PyTorch 2.6 + CUDA 12.4, FP32.

### Hallucination in descriptions (CHAIR)

100 COCO val2017 images, one caption each (`tools/eval_hallucination.py --task chair`).
CHAIR_s is the share of captions that name at least one object that is not in the image;
CHAIR_i is the share of all named objects that are not there; recall is the share of
real objects the captions cover. Lower is better for the first two.

| Method (language model: SmolVLM2-500M) | CHAIR_s | CHAIR_i | Recall | Words |
|----------------------------------------|---------|---------|--------|-------|
| First version: detector labels given to the model as facts | 31.0 % | 11.6 % | 61.0 % | 24.6 |
| Language model alone | 20.0 % | 9.3 % | 52.8 % | 20.4 |
| Verify, lenient: cut only if the detector and the model's own yes/no both say no | 17.0 % | 8.1 % | 52.8 % | 20.2 |
| **Verify, strict (default): cut whatever the detector cannot confirm** | **5.0 %** | **3.2 %** | **49.1 %** | 13.6 |

What this shows:

- Feeding detector output to the model as facts raised the share of captions with an
  invented object from 20 % to 31 %. The model repeated the detector's mistakes.
- Strict verification cut it to 5 %, six times lower than the first version, for a
  recall cost of about 4 points against the model alone. Captions also get shorter,
  because a sentence with an unconfirmed object is removed whole.
- The lenient policy barely helps with this model: asked directly, it stands by almost
  everything it said (it rejected 3 objects in 100 images).
- With 100 images the 31 % vs 5 % gap is far outside sampling noise; the 20 % vs 17 %
  gap is not.

### Hallucination in yes/no questions (POPE)

100 COCO images, 1,200 questions of the form "Is there a ... in the image?", half about
objects that are present (`tools/eval_hallucination.py --task pope`). Negatives are
chosen at random, from the most frequent categories, or from categories that usually
co-occur with what is present (adversarial).

| Setting | Language model alone: accuracy / F1 / says yes | First version (detector facts in prompt): accuracy / F1 / says yes |
|---------|---------------------------|----------------------------|
| Random | 88.3 % / 0.884 / 51 % | 81.7 % / 0.778 / 33 % |
| Popular | 78.3 % / 0.804 / 61 % | 79.8 % / 0.761 / 34 % |
| Adversarial | 72.8 % / 0.766 / 66 % | 78.5 % / 0.750 / 36 % |

The small model alone says "yes" too easily to plausible objects (adversarial accuracy
73 %). Adding the detector's list made it cautious instead: fewer false alarms, but it
then missed a third of the objects that really were there (recall 64 % against 89 %).
Questions are now answered from the image plus measured positions and distances of
confirmed objects only. A stronger language model is the real fix here, which is what
the Colab profiles are for.

### Events on the two sample clips

| Clip | Events, first version | Events, now |
|------|-----------------------|-------------|
| Dog in a square, 11 s | 49 | 16 |
| Two people on a sofa, 16 s | 88 | 28 |

The first version announced every detector label, including a hand cart labelled
"truck". Now only labels confirmed by both models produce events, and a tracker id
switch on a still-visible object is re-linked instead of reported as "left" then
"appeared".

### Speed

4K 25 fps video resized to 960 px wide; timings use `torch.cuda.synchronize()`.

| Stage | Median |
|-------|--------|
| Detect + segment (YOLOE-26s, 640, masks at frame resolution) | 40 ms |
| Track | 3 ms |
| Depth (every 3rd frame) | 49 ms |
| Scene memory | 1 ms |
| **End to end, perception only** | **70 ms, 12.3 FPS** |
| End to end with the language loop running | about 150 to 175 ms, 5 to 6 FPS |
| One description (two model calls plus verification), concurrent with perception | about 10 s |
| One answer to a question | 0.7 to 1.9 s |
| Changing the detector vocabulary | 1 ms (plus about 1 s per new word, off the camera thread) |
| Peak GPU memory | 2.5 GB |

Plainly: perception is interactive, and a description arrives roughly ten seconds
after the change that triggered it on this GPU. Half precision does not help on this
card (`tools/precision_check.py`: 29 ms in FP32, 86 ms in FP16 for the depth model).

### Detection accuracy

COCO val2017, first 1,000 images, standard COCO box mAP (`tools/eval_coco.py`).
YOLOE is prompted with the 80 class names as text and was not trained on COCO, so its
rows are zero-shot. YOLOv8m is the supervised closed-set detector from the earlier
version of this project, run on the same images.

| Detector | Vocabulary | mAP 50-95 | mAP 50 | Small / medium / large |
|----------|------------|-----------|--------|------------------------|
| YOLOv8m (earlier project) | 80 fixed classes, trained on COCO | 0.526 | 0.694 | 0.342 / 0.590 / 0.655 |
| YOLOE-26s (`laptop` profile) | any text, zero-shot | 0.383 | 0.537 | 0.224 / 0.427 / 0.521 |
| YOLOE-26m (`--weights yoloe-26m-seg.pt`) | any text, zero-shot | 0.438 | 0.605 | 0.287 / 0.477 / 0.567 |

On the 80 classes COCO was built around, the supervised detector is more accurate. The
open-vocabulary detector is what makes verification of arbitrary objects possible at
all, and its accuracy is the ceiling on how much real content strict verification keeps.

Not measured yet: the Colab profiles, the full 5,000-image set, rare categories (LVIS),
tracking accuracy, depth error, and answer quality beyond yes/no questions.

## Known limitations

- Strict verification also removes true statements when the detector misses a real
  object. Example from the benchmark: a close-up of a bear, described correctly, was
  cut because the detector found only "eye" and "nose".
- When both models make the same mistake it passes. On the sofa clip the game
  controllers are still sometimes called phones, because the detector's "phone" and the
  small language model's "phone" agree.
- Only objects are verified. Colours, actions and relations in a description are not.
- With the 500M language model, answers are short and descriptions are simple.
- People in dark, close-up footage are sometimes missed by the small detector.
- Metric depth comes from an indoor-trained model; outdoor distances beyond about
  10 m are rough.

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
.\.venv\Scripts\python.exe run.py --profile laptop                 # force a profile (default: auto)
```

**Asking questions.** Click the video window, press `a`, type your question in the
window and press Enter (Esc cancels). A video file pauses while you type. The answer
appears in green above the description. Other keys: `space` pauses, `d` forces a fresh
description, `q` quits.

Without a window, ask from the command line:

```powershell
.\.venv\Scripts\python.exe run.py --source samples\dog_square.mp4 --headless --ask "What is the dog doing?" --ask-at 3
```

**Without a window** (save an annotated video and a latency report):

```powershell
.\.venv\Scripts\python.exe run.py --source samples\dog_square.mp4 --headless --save outputs\demo.mp4 --report outputs\bench.json
```

**Tests and evaluation:**

```powershell
.\.venv\Scripts\python.exe tests\test_memory.py            # unit tests, no GPU needed
.\.venv\Scripts\python.exe tests\test_describe.py
.\.venv\Scripts\python.exe tests\test_chair.py
.\.venv\Scripts\python.exe tools\eval_coco.py --n 1000     # detection mAP on a COCO subset
.\.venv\Scripts\python.exe tools\eval_hallucination.py --vlm HuggingFaceTB/SmolVLM2-500M-Video-Instruct --task chair --mode verified --two-step
.\.venv\Scripts\python.exe tools\eval_hallucination.py --vlm HuggingFaceTB/SmolVLM2-500M-Video-Instruct --task pope
.\.venv\Scripts\python.exe tools\precision_check.py samples\dog_square.mp4   # FP32 vs FP16 on your GPU
```

In VS Code, open the project folder and pick `.venv` as the Python interpreter
(Ctrl+Shift+P, "Python: Select Interpreter"); then `run.py` runs with the Run button.

The `samples` folder is not part of the repository. Use any video of your own, or a
webcam.

## Run on Google Colab

Colab is where the full-quality profiles run: a 3B or 7B language model, the largest
detector and depth model. It cannot use your webcam at interactive speed, so it works
on video files. The notebook
[notebooks/livescene_colab.ipynb](notebooks/livescene_colab.ipynb) contains every step
below as runnable cells. It has not been run on Colab yet; if a cell fails, the error
message is what is needed to fix it.

1. Open [colab.research.google.com](https://colab.research.google.com), choose
   File, Upload notebook, and upload `notebooks/livescene_colab.ipynb`.
2. Runtime, Change runtime type, select a GPU, Save. A T4 selects the `colab-t4`
   profile; an L4 or A100 selects `colab`.
3. Get the code into Colab. If the repository is public:

   ```
   !git clone https://github.com/HameesSajid/livescene.git
   %cd livescene
   ```

   If it is private: on GitHub choose Code, Download ZIP, set `USE_ZIP = True` in the
   notebook's second cell, and upload the ZIP when asked.
4. Install the libraries (PyTorch with CUDA is already on Colab):

   ```
   !pip -q install -r requirements.txt
   !pip -q install https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
   ```
5. Upload a video, process it, and ask it a question:

   ```
   from google.colab import files
   video = next(iter(files.upload()))
   !python run.py --source "$video" --headless --save outputs/colab_demo.mp4 --report outputs/colab_bench.json --ask "What is happening in this video?" --ask-at 3
   ```

   The first run downloads the models (about 8 GB for `colab-t4`, 18 GB for `colab`).
6. Watch the result (re-encoded so the browser can play it):

   ```
   !ffmpeg -y -loglevel error -i outputs/colab_demo.mp4 -vcodec libx264 outputs/colab_demo_h264.mp4
   from IPython.display import Video
   Video("outputs/colab_demo_h264.mp4", embed=True, width=800)
   ```
7. Measure hallucination for the model your GPU selected (this is the table to compare
   with the laptop numbers above):

   ```
   !python tools/eval_hallucination.py --vlm "$VLM" --task pope --images 200 --max-side $SIDE --out outputs/pope_colab_plain.json
   !python tools/eval_hallucination.py --vlm "$VLM" --task chair --mode plain --images 200 --tokens 120 --max-side $SIDE --out outputs/chair_colab_plain.json
   !python tools/eval_hallucination.py --vlm "$VLM" --task chair --mode verified --detector $DETECTOR --images 200 --tokens 120 --max-side $SIDE --out outputs/chair_colab_verified.json
   ```
8. Run the full COCO detection evaluation (all 5,000 validation images):

   ```
   !python tools/eval_coco.py --n 5000 --weights $DETECTOR --out outputs/coco_colab_full.json
   ```
9. Download the JSON files and the video from `outputs/` with the last cell. Colab
   deletes the session's files when it disconnects.

## Local or Colab?

| | Local GPU (6 GB) | Colab |
|-|------------------|-------|
| Live webcam, questions typed in the window | Yes | No |
| Video files, questions with `--ask` | Yes | Yes |
| Language model | 500M parameters | 3B or 7B parameters |
| Detector and depth model | Small | Large |
| Several frames per description (motion) | No | Yes |
| Full benchmarks | Slow but possible | Recommended |

## Roadmap

1. Run the Colab profiles and add their hallucination and accuracy numbers.
2. Verify attributes and actions, not only objects.
3. Stop double errors: require the verifying detector to beat competing labels, not
   only to find the object.
4. TensorRT export of the detector and depth model for the laptop profile.
5. Pose keypoints for action events.

## Repository layout

```
run.py                       entry point: window, keys, questions, reports
livescene/config.py          model profiles (laptop, colab-t4, colab)
livescene/describe.py        propose, verify, describe; prompts and text checks
livescene/vlm.py             language model backends
livescene/perception.py      detector with a changeable vocabulary, tracker, depth
livescene/memory.py          scene memory: trust, events, re-linking
livescene/narrator.py        language thread
livescene/pipeline.py        the two loops and the overlay
livescene/profiler.py        latency statistics
livescene/paths.py           where downloaded models and data are stored
tests/                       27 unit tests, no GPU needed
tools/eval_hallucination.py  POPE and CHAIR benchmarks
tools/chair.py               CHAIR scorer
tools/eval_coco.py           COCO detection mAP
tools/precision_check.py     FP32 vs FP16 check for the current GPU
notebooks/                   Colab notebook
docs/                        concepts, code guide, install log
outputs/                     benchmark reports (annotated videos are git-ignored)
```
## Credits

Built on [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/),
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),
[SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and ByteTrack. The benchmarks follow POPE
(Li et al., 2023) and CHAIR (Rohrbach et al., 2018). Check each model's licence before commercial use (Ultralytics is AGPL-3.0).
