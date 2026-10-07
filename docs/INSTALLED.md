# What was installed, where, and how to remove it

Last updated: 2026-10-07. Sizes are as measured on that date.

Nothing was added to your existing system Python (3.10, CPU-only PyTorch). It is
unchanged. LiveScene uses its own environment, which is the one that uses the GPU.

## 1. The LiveScene folder (everything heavy)

`C:\Users\hamee\cv-envs\livescene\` — about 8.1 GB, outside OneDrive on purpose so it
does not sync.

| Path | Size | Contents |
|------|------|----------|
| `venv\` | 5.1 GB | Python virtual environment: PyTorch 2.6.0 + CUDA 12.4, torchvision 0.21.0, ultralytics 8.4.174, transformers 5.19.0, accelerate, lap, pycocotools, opencv-python 5.0.0, numpy 2.2.6, pillow, num2words, clip (Ultralytics fork) and their dependencies |
| `hf\hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct\` | 1.9 GB | Vision-language model |
| `hf\hub\models--depth-anything--Depth-Anything-V2-Metric-Indoor-Small-hf\` | 95 MB | Depth model |
| `weights\yoloe-26s-seg.pt` | 30 MB | Detector in use |
| `weights\mobileclip2_b.ts` | 242 MB | Text encoder for the detector in use |
| `weights\yoloe-26m-seg.pt` | 67 MB | Medium detector, the higher-accuracy option (`--weights yoloe-26m-seg.pt`) |
| `weights\yolov8m.pt` | 50 MB | The earlier project's detector, used only as the baseline in `tools/eval_coco.py` |
| `weights\yoloe-11s-seg.pt`, `yoloe-11m-seg.pt` | 84 MB | Detectors tried during model selection; not used, safe to delete |
| `weights\mobileclip_blt.ts` | 572 MB | Text encoder for the YOLOE-11 models; not used, safe to delete |
| `data\coco\instances_val2017.json` | 20 MB | COCO val2017 annotations, for `tools/eval_coco.py` |
| `data\coco\val2017\` | 154 MB | The first 1,000 COCO val2017 images |

Remove everything:

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\cv-envs\livescene"
```

Remove only the unused experiment weights (frees about 650 MB):

```powershell
cd "$env:USERPROFILE\cv-envs\livescene\weights"
Remove-Item yoloe-11s-seg.pt, yoloe-11m-seg.pt, mobileclip_blt.ts
```

Remove only the COCO evaluation data (frees about 175 MB; it re-downloads if you run
the evaluation again):

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\cv-envs\livescene\data"
```

## 2. Small files outside that folder

| Path | Size | What | Remove with |
|------|------|------|-------------|
| `%APPDATA%\Ultralytics\` | a few KB | Ultralytics settings file, created on first import | `Remove-Item -Recurse "$env:APPDATA\Ultralytics"` |
| `%LOCALAPPDATA%\pip\cache\` | 3.7 GB total | pip's download cache. Shared with your other Python work; the LiveScene install added roughly 3 GB of wheels to it | `python -m pip cache purge` (safe; only makes future installs re-download) |

## 3. Inside the project folder (synced by OneDrive)

`...\Computer vision\project\livescene\` — source code, docs, the hidden `.git`
folder, and `outputs\` (two annotated demo videos, about 5 MB each, plus JSON
benchmark and accuracy reports). No model weights, datasets or environments are
stored here. The same content, minus the videos, is on GitHub at
https://github.com/HameesSajid/livescene.

## 4. Using the GPU instead of the CPU

Run LiveScene with the environment's own interpreter. That is the only step; the code
selects CUDA automatically when it is available.

```powershell
& "$env:USERPROFILE\cv-envs\livescene\venv\Scripts\python.exe" run.py
```

Check that the GPU is in use:

```powershell
& "$env:USERPROFILE\cv-envs\livescene\venv\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Running `python run.py` with your system Python would fail, because that Python has
CPU-only PyTorch and none of the other libraries.

## 5. Git

Git for Windows 2.56.0 was installed by you (not by the LiveScene setup) at
`C:\Program Files\Git\`. Remove it from Windows Settings, Apps, Installed apps if you
no longer want it. The repository's own history lives in the hidden `livescene\.git`
folder; the name and email used for commits are set for this repository only, not
globally. A saved GitHub sign-in, once you approve one, is kept in Windows Credential
Manager under `git:https://github.com` and can be removed there.

The GitHub CLI is not installed.

## Change log

| Date | Change |
|------|--------|
| 2026-10-07 | Created `cv-envs\livescene\venv`; installed PyTorch CUDA, ultralytics, transformers, accelerate, lap, pycocotools, num2words, Ultralytics CLIP fork |
| 2026-10-07 | Downloaded SmolVLM2-500M, Depth Anything V2 Metric Indoor Small, YOLOE-26s and its MobileCLIP2 text encoder |
| 2026-10-07 | Downloaded YOLOE-11s/11m/26m and MobileCLIP-BLT for comparison; left in place, unused |
| 2026-10-07 | Git for Windows 2.56.0 installed by the user; repository initialised in `livescene\` |
| 2026-10-07 | Downloaded COCO val2017 annotations and 1,000 images to `data\coco`; downloaded `yolov8m.pt` as the evaluation baseline |
