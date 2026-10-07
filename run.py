"""LiveScene entry point.

Live webcam:        python run.py
Video file:         python run.py --source path/to/video.mp4
Custom vocabulary:  python run.py --vocab "person,coffee mug,red backpack"
Benchmark, no UI:   python run.py --source video.mp4 --headless --max-frames 300 --report out.json

Keys in the window: q quit, d describe now, a ask a question (typed in the console).
"""

import argparse
import json
import os
import threading
import time

from livescene.paths import CACHE  # sets HF_HOME; keep above the model libraries

import cv2  # noqa: E402
import torch  # noqa: E402

from livescene.pipeline import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--vocab", default=None, help="comma-separated object names to detect")
    ap.add_argument("--width", type=int, default=960, help="frames are resized to this width")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--weights", default="yoloe-26s-seg.pt")
    ap.add_argument("--depth-every", type=int, default=3)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-vlm", action="store_true")
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

    cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW) if live else cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read from source {args.source!r}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scale = args.width / frame.shape[1]
    size = (args.width, int(round(frame.shape[0] * scale / 2) * 2))

    vocab = [v.strip() for v in args.vocab.split(",") if v.strip()] if args.vocab else None
    pipe = Pipeline(size, vocab, use_depth=not args.no_depth, use_vlm=not args.no_vlm,
                    depth_every=args.depth_every, det_weights=args.weights, imgsz=args.imgsz, conf=args.conf)
    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, size) if args.save else None

    questions = []
    if live and not args.headless:
        def read_questions():
            while True:
                questions.append(input())
        threading.Thread(target=read_questions, daemon=True).start()

    n, fps, t_start, warmup = 0, 0.0, time.perf_counter(), 10
    while ok and (not args.max_frames or n < args.max_frames):
        t0 = time.perf_counter()
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        # Live: wall clock. File: the video's own timeline, so events are reproducible.
        t = (t0 - t_start) if live else n / src_fps
        for ev in pipe.step(frame, t):
            print(f"[{t:7.2f}s] {ev.kind:8s} {ev.text}")
        if questions:
            pipe.ask(frame, t, questions.pop(0))
        if writer is not None or not args.headless:
            with pipe.prof.stage("draw"):
                vis = pipe.draw(frame, fps)
            if writer is not None:
                writer.write(vis)
            if not args.headless:
                cv2.imshow("LiveScene", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    pipe.memory.pending_salience += 10
                if key == ord("a"):
                    print("type your question and press Enter:")
        dt = time.perf_counter() - t0
        if n >= warmup:
            pipe.prof.add("end-to-end", dt * 1000.0)
        fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
        n += 1
        with pipe.prof.stage("capture (decode)"):
            ok, frame = cap.read()

    if pipe.narrator is not None:  # let an in-flight narration finish so it is logged
        deadline = time.time() + 30
        while pipe.narrator.busy and time.time() < deadline:
            time.sleep(0.05)
        pipe._collect_narration(n / src_fps)
    summary = pipe.prof.summary()
    e2e = summary.get("end-to-end", {}).get("mean_ms")
    print("\nstage                      mean ms   p50 ms   p95 ms      n")
    for name, s in summary.items():
        print(f"{name:25s} {s['mean_ms']:8.1f} {s['p50_ms']:8.1f} {s['p95_ms']:8.1f} {s['n']:6d}")
    if e2e:
        print(f"\nend-to-end: {1000.0 / e2e:.1f} FPS over {n} frames; {pipe.narration_count} narrations")
    if args.report:
        report = {
            "source": os.path.basename(args.source), "frames": n, "frame_size": size,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "fps": 1000.0 / e2e if e2e else None, "latency": summary,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
            "narrations": pipe.narration_count,
            "log": [{"t": round(t, 2), "kind": k, "text": x} for t, k, x in pipe.log],
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
