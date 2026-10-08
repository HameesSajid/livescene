# What was installed, where, and how to remove it

Last updated: 2026-10-08. Sizes are as measured on that date.

Nothing was added to your existing system Python (3.10, CPU-only PyTorch). It is
unchanged. LiveScene uses its own environment, which is the one that uses the GPU.

## 1. Everything is in one folder

`C:\Users\hamee\vscode\livescene\` holds the whole project: source code, documents,
the Python environment, model weights and the evaluation dataset. About 9 GB in
total. It is not inside OneDrive, so none of it syncs.

| Path inside the project folder | Size | Contents | On GitHub? |
|--------------------------------|------|----------|------------|
| `run.py`, `livescene\`, `tests\`, `tools\`, `notebooks\` | under 1 MB | Source code and the Colab notebook | Yes |
| `README.md`, `docs\` | under 1 MB | Documentation | Yes |
| `outputs\*.json` | under 1 MB | Benchmark and accuracy reports | Yes |
| `outputs\*.mp4` | 12 MB | Two annotated demo videos | No |
| `samples\` | 104 MB | Copies of your two test clips (`dog_square.mp4`, `gaming_sofa.mp4`). The originals are still in your OneDrive project folder | No |
| `.venv\` | 5.1 GB | Python virtual environment, 66 packages. Installed directly: torch 2.6.0+cu124, torchvision 0.21.0+cu124, ultralytics 8.4.174, transformers 5.19.0, accelerate 1.15.0, num2words 0.5.14, lap 0.5.13, pycocotools 2.0.11, opencv-python 5.0.0.93, numpy 2.2.6, pillow 12.3.0, clip 1.0 (Ultralytics fork). The rest are their dependencies | No |
| `.cache\hf\hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct\` | 1.9 GB | Vision-language model | No |
| `.cache\hf\hub\models--depth-anything--Depth-Anything-V2-Metric-Indoor-Small-hf\` | 95 MB | Depth model | No |
| `.cache\weights\yoloe-26s-seg.pt` | 30 MB | Detector in use | No |
| `.cache\weights\mobileclip2_b.ts` | 242 MB | Text encoder for the YOLOE-26 detectors | No |
| `.cache\weights\yoloe-26m-seg.pt` | 67 MB | Medium detector, the higher-accuracy option | No |
| `.cache\weights\yolov8m.pt` | 50 MB | The earlier project's detector, used only as the baseline in `tools\eval_coco.py` | No |
| `.cache\weights\yoloe-11s-seg.pt`, `yoloe-11m-seg.pt` | 84 MB | Detectors tried during model selection; not used, safe to delete | No |
| `.cache\weights\mobileclip_blt.ts` | 572 MB | Text encoder for the YOLOE-11 models; not used, safe to delete | No |
| `.cache\data\coco\` | 178 MB | COCO val2017 annotations (instances and captions) and the first 1,000 images, for the benchmarks | No |
| `.cache\data\chair\synonyms.txt` | 5 KB | The official CHAIR synonym list | No |
| `.cache\eval\` | under 1 MB | Per-image benchmark logs (every caption and answer) | No |
| `.git\` (hidden) | under 1 MB | Git history | It is the link to GitHub |

The "No" rows are excluded by `.gitignore`. Anything in `.cache` re-downloads
automatically the next time it is needed.

### Removing things

Remove everything (the code stays on GitHub):

```powershell
Remove-Item -Recurse -Force C:\Users\hamee\vscode\livescene
```

Remove only the downloaded models and data, keeping code and environment (frees
about 3.3 GB):

```powershell
Remove-Item -Recurse -Force C:\Users\hamee\vscode\livescene\.cache
```

Remove only the unused experiment weights (frees about 650 MB):

```powershell
cd C:\Users\hamee\vscode\livescene\.cache\weights
Remove-Item yoloe-11s-seg.pt, yoloe-11m-seg.pt, mobileclip_blt.ts
```

Remove only the Python environment (frees about 5.1 GB; rebuild it with the setup
steps in the README):

```powershell
Remove-Item -Recurse -Force C:\Users\hamee\vscode\livescene\.venv
```

## 2. Small files outside the project folder

| Path | Size | What | Remove with |
|------|------|------|-------------|
| `%APPDATA%\Ultralytics\` | a few KB | Ultralytics settings file, created on first import | `Remove-Item -Recurse "$env:APPDATA\Ultralytics"` |
| `%LOCALAPPDATA%\pip\cache\` | 3.8 GB total | pip's download cache. Shared with your other Python work; the LiveScene installs added roughly 3 GB of downloaded packages to it, including the bitsandbytes download described in section 6 | `python -m pip cache purge` (safe; only makes future installs re-download) |
| Windows Credential Manager, entry `git:https://github.com` | none | Your saved GitHub sign-in for pushing | Control Panel, Credential Manager, Windows Credentials |

## 3. Using the GPU instead of the CPU

Run LiveScene with the environment's own interpreter. That is the only step; the code
selects CUDA automatically when it is available.

```powershell
cd C:\Users\hamee\vscode\livescene
.\.venv\Scripts\python.exe run.py
```

Check that the GPU is in use:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Running plain `python run.py` would use your system Python, which has CPU-only
PyTorch and none of the other libraries, and would fail.

## 4. Git and GitHub

Git for Windows 2.56.0 was installed by you (not by the LiveScene setup) at
`C:\Program Files\Git\`. Remove it from Windows Settings, Apps, Installed apps if you
no longer want it. The name and email used for commits are set for this repository
only, not globally. The repository is https://github.com/HameesSajid/livescene.

The GitHub CLI is not installed.

## 5. Previous locations (now empty)

| Old location | Status |
|--------------|--------|
| `...\OneDrive - UAB...\Computer vision\project\livescene\` | Moved to `C:\Users\hamee\vscode\livescene\` on 2026-10-07. Your original project files in that OneDrive folder were not touched |
| `C:\Users\hamee\cv-envs\livescene\` | Models and data moved into `.cache`; the environment was rebuilt as `.venv` and the old copy deleted |

## 6. Deliberately not installed on this PC

At your request (2026-10-08), nothing further is installed or downloaded here for the
larger models. In particular:

| Item | Size it would take | Where it is used instead |
|------|--------------------|--------------------------|
| `bitsandbytes` (4-bit model loading) | 0.1 GB | Not needed; the Colab profiles use half precision |
| Qwen2.5-VL-3B / 7B language models | 7.5 GB / 17 GB | Colab only (`colab-t4` and `colab` profiles) |
| YOLOE-26l / 26x detectors | 76 MB / 164 MB | Colab only |
| Depth Anything V2 Base / Large | 0.4 GB / 1.3 GB | Colab only |

One correction for the record: `bitsandbytes` 0.50.2 was installed into `.venv` on
2026-10-08, because the install had already finished when the command it was part of
was stopped. It was uninstalled the same day (`pip uninstall bitsandbytes`). It
brought no other packages with it, and no Qwen files were downloaded. Its downloaded
package file may remain in pip's cache (section 2).

Running `--profile colab` or `--profile colab-t4` on this PC would start downloading
those models into `.cache`. The default `auto` never selects them on a 6 GB GPU.

## Change log

| Date | Change |
|------|--------|
| 2026-10-07 | Created the Python environment; installed PyTorch CUDA, ultralytics, transformers, accelerate, lap, pycocotools, num2words, Ultralytics CLIP fork |
| 2026-10-07 | Downloaded SmolVLM2-500M, Depth Anything V2 Metric Indoor Small, YOLOE-26s and its MobileCLIP2 text encoder |
| 2026-10-07 | Downloaded YOLOE-11s/11m/26m and MobileCLIP-BLT for comparison |
| 2026-10-07 | Git for Windows 2.56.0 installed by the user; repository initialised and pushed to GitHub |
| 2026-10-07 | Downloaded COCO val2017 annotations and 1,000 images; downloaded `yolov8m.pt` as the evaluation baseline |
| 2026-10-07 | Moved the whole project to `C:\Users\hamee\vscode\livescene\`; environment rebuilt there as `.venv`, downloads moved to `.cache`, sample clips copied to `samples\` |
| 2026-10-08 | Downloaded COCO reference captions (into `.cache\data\coco`) and the CHAIR synonym list (into `.cache\data\chair`) for the hallucination benchmarks |
| 2026-10-08 | `bitsandbytes` 0.50.2 installed into `.venv`, then uninstalled the same day at the user's request. Environment is back to 66 packages |
| 2026-10-08 | Decision recorded: no further installs or model downloads on this PC; larger models run on Colab only |
