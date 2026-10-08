"""LiveScene entry point.

Live webcam:        python run.py
Video file:         python run.py --source path/to/video.mp4
Ask about a video:  python run.py --source video.mp4 --ask "What is the dog doing?" --ask-at 3
Benchmark, no UI:   python run.py --source video.mp4 --headless --save out.mp4 --report out.json
Bigger models:      python run.py --profile colab        (default: chosen from GPU memory)

Keys in the window:
  a      ask a question: type it in the window, Enter sends, Esc cancels
  space  pause / resume
  d      describe the scene now
  q      quit
"""

import argparse
import json
import os
import time

from livescene.paths import CACHE  # sets HF_HOME; keep above the model libraries

import cv2  # noqa: E402
import torch  # noqa: E402

from livescene.config import PROFILES, pick  # noqa: E402
from livescene.pipeline import Pipeline  # noqa: E402

ENTER, ESC, BACKSPACE, SPACE = 13, 27, 8, 32


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--profile", default="auto", choices=["auto"] + list(PROFILES))
    ap.add_argument("--vlm", default=None, help="override the profile's language model")
    ap.add_argument("--weights", default=None, help="override the profile's detector weights")
    ap.add_argument("--vocab", default=None, help="comma-separated starting vocabulary for the detector")
    ap.add_argument("--width", type=int, default=None, help="frames are resized to this width")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--vlm-frames", type=int, default=None, help="frames per language-model call")
    ap.add_argument("--depth-every", type=int, default=3)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--ask", action="append", default=[], help="a question to ask (repeatable)")
    ap.add_argument("--ask-at", type=float, default=2.0, help="seconds into the stream to ask at")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save", default=None, help="write the annotated video here")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--report", default=None, help="write latency + event log JSON here")
    args = ap.parse_args()

    live = args.source.isdigit()
    for name in ("save", "report") + (() if live else ("source",)):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    # Ultralytics downloads weights into the working directory, so work from the cache.
    os.makedirs(os.path.join(CACHE, "weights"), exist_ok=True)
    os.chdir(os.path.join(CACHE, "weights"))

    cfg = pick(args.profile, vlm=args.vlm, detector=args.weights, verifier=args.weights, width=args.width,
               imgsz=args.imgsz, vlm_frames=args.vlm_frames)
    print(f"profile: {cfg.name} | detector {cfg.detector} | language model {'off' if args.no_vlm else cfg.vlm}")

    cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW) if live else cv2.VideoCapture(args.source)
    ok, raw = cap.read()
    if not ok:
        raise SystemExit(f"cannot read from source {args.source!r}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scale = cfg.width / raw.shape[1]
    size = (cfg.width, int(round(raw.shape[0] * scale / 2) * 2))

    vocab = [v.strip() for v in args.vocab.split(",") if v.strip()] if args.vocab else None
    # A file is processed frame by frame, so the tracker sees the video's own frame rate.
    pipe = Pipeline(size, cfg, vocab, use_depth=not args.no_depth, use_vlm=not args.no_vlm,
                    depth_every=args.depth_every, fps_hint=10.0 if live else src_fps)
    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, size) if args.save else None
    asks = [(args.ask_at + 4.0 * i, q) for i, q in enumerate(args.ask)]

    n, fps, t_start, warmup = 0, 0.0, time.perf_counter(), 10
    frame, t, paused, typing, typing_paused, printed = None, 0.0, False, None, False, 0
    while True:
        t0 = time.perf_counter()
        stepped = False
        if not paused:
            if not ok or (args.max_frames and n >= args.max_frames):
                break
            frame = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
            # Live: wall clock. File: the video's own timeline, so events are reproducible.
            t = (t0 - t_start) if live else n / src_fps
            pipe.step(frame, t)
            n += 1
            stepped = True
        elif pipe.narrator is not None:
            pipe.poll(t)
        while asks and t >= asks[0][0]:
            pipe.ask(frame, t, asks.pop(0)[1])
        for et, kind, text in pipe.log[printed:]:
            print(f"[{et:7.2f}s] {kind:11s} {text}")
        printed = len(pipe.log)

        if writer is not None or not args.headless:
            with pipe.prof.stage("draw"):
                vis = pipe.draw(frame, fps, typing, paused)
            if writer is not None and stepped:
                writer.write(vis)
            if not args.headless:
                cv2.imshow("LiveScene", vis)
                quit_now = False
                for _ in range(12):  # drain queued keystrokes so typing keeps up at low frame rates
                    key = cv2.waitKey(25 if paused else 1)
                    if key < 0:
                        break
                    key &= 0xFF
                    if typing is not None:
                        if key == ENTER:
                            if typing.strip():
                                pipe.ask(frame, t, typing.strip())
                            typing, paused, typing_paused = None, paused and not typing_paused, False
                        elif key == ESC:
                            typing, paused, typing_paused = None, paused and not typing_paused, False
                        elif key == BACKSPACE:
                            typing = typing[:-1]
                        elif 32 <= key < 127:
                            typing += chr(key)
                    elif key == ord("q"):
                        quit_now = True
                    elif key == ord("a") and pipe.narrator is not None:
                        typing = ""
                        if not live and not paused:  # hold the video still while the question is typed
                            paused = typing_paused = True
                    elif key == SPACE:
                        paused = not paused
                    elif key == ord("d"):
                        pipe.memory.pending_salience += 10
                if quit_now:
                    break
        if stepped:
            dt = time.perf_counter() - t0
            if n > warmup:
                pipe.prof.add("end-to-end", dt * 1000.0)
            fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
            with pipe.prof.stage("capture (decode)"):
                ok, raw = cap.read()

    if pipe.narrator is not None:  # let queued and in-flight language jobs finish so they are logged
        deadline = time.time() + 120
        while (pipe.narrator.busy or pipe.narrator.question_waiting) and time.time() < deadline:
            time.sleep(0.05)
        pipe.poll(t)
        for et, kind, text in pipe.log[printed:]:
            print(f"[{et:7.2f}s] {kind:11s} {text}")
    summary = pipe.prof.summary()
    e2e = summary.get("end-to-end", {}).get("mean_ms")
    print("\nstage                      mean ms   p50 ms   p95 ms      n")
    for name, s in summary.items():
        print(f"{name:25s} {s['mean_ms']:8.1f} {s['p50_ms']:8.1f} {s['p95_ms']:8.1f} {s['n']:6d}")
    if e2e:
        print(f"\nend-to-end: {1000.0 / e2e:.1f} FPS over {n} frames; {pipe.description_count} descriptions")
    if args.report:
        report = {
            "source": os.path.basename(args.source), "frames": n, "frame_size": size, "profile": cfg.name,
            "detector": cfg.detector, "language_model": None if args.no_vlm else cfg.vlm,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "fps": 1000.0 / e2e if e2e else None, "latency": summary,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
            "descriptions": pipe.description_count, "final_vocabulary_added": list(pipe.extra_vocab),
            "log": [{"t": round(et, 2), "kind": k, "text": x} for et, k, x in pipe.log],
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    pipe.close()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
