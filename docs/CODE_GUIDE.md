# Code guide

A file-by-file, function-by-function explanation of LiveScene. Read
[CONCEPTS.md](CONCEPTS.md) first for the ideas; this document is about the code.

## The life of one frame

1. `run.py` reads a frame and resizes it to 960 px wide.
2. `Pipeline.step` sends it to `OpenVocabTracker`, which returns boxes, labels,
   confidences, track IDs and mask polygons.
3. On every third frame, `MetricDepth` produces a small depth map in metres. Other
   frames reuse the last one.
4. `to_detections` computes one distance per object (median depth inside its mask)
   and wraps everything as `Detection` objects.
5. `SceneMemory.update` merges the detections into its tracks and returns any events.
6. If `SceneMemory.should_narrate` says enough has changed and the narrator is free,
   the frame and `SceneMemory.grounding_prompt` go to the `Narrator` thread.
7. `Pipeline.draw` renders masks, boxes, the event ticker and the latest narration.

Steps 2 to 5 and 7 run on the main thread. Step 6 only hands work off; the language
model generates on its own thread and its answer is picked up on a later frame.

---

## `livescene/memory.py` — the scene memory

Pure Python and NumPy. No GPU, no model. This is where per-frame detections become a
stable description, and it is the part covered by unit tests.

### Data classes

- **`Detection`** — one object in one frame: `track_id`, `label`, `conf`, `box`
  (x1, y1, x2, y2 in pixels), `distance` (metres, or `None` without depth).
- **`Event`** — something that happened: time `t`, `kind` (`enter`, `exit`,
  `approach`, `recede`, `near`), human-readable `text`, the `track_ids` involved,
  and `salience` (how much it counts towards triggering narration).
- **`Track`** — the memory's record of one object across frames. Holds first and last
  seen time, `hits` (frames seen), `confirmed` flag, latest box and smoothed distance,
  `labels` (a confidence-weighted vote counter), `dist_hist` (recent
  `(time, distance)` pairs), and a cooldown timestamp for motion events.
  - `center` — box centre in pixels.
  - `name` — display name such as `dog #2`.

### Helper functions

- **`_box_gap(a, b)`** — pixel distance between two boxes; 0 if they touch or overlap.
  Used to decide whether two objects are adjacent in the image.
- **`_plural(label, n)`** — "1 cup", "2 cups", "3 people", "2 boxes".

### `SceneMemory`

Constructor parameters (all thresholds live here so they can be tuned in one place):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `frame_size` | required | (width, height), used for left / centre / right and the adjacency threshold |
| `confirm_hits` | 3 | Frames an object must be seen before it is announced. Filters one-frame false detections |
| `exit_after_s` | 1.0 | Seconds unseen before an object is declared gone. Bridges short detection gaps |
| `motion_window_s` | 1.5 | Window over which distance change is measured |
| `motion_rel_change` | 0.25 | Fractional distance change that counts as approaching or receding |
| `motion_cooldown_s` | 3.0 | Minimum gap between motion events for the same object |
| `near_gap_frac` | 0.03 | Adjacency threshold as a fraction of the frame diagonal |
| `near_depth_m` | 0.6 | Maximum depth difference for two objects to count as next to each other |
| `near_hold_s` | 0.5 | Adjacency must persist this long before it is reported |
| `near_release_s` | 1.5 | Adjacency is forgotten only after this long apart |

Methods:

- **`update(t, detections)`** — the main entry point, called once per frame. For each
  detection it creates or refreshes the track, adds the label to the vote (so a
  track that is called "cup" 25 times and "mug" 5 times stays "cup"), smooths the
  distance with an exponential moving average (0.7 old, 0.3 new), and emits `enter`
  when the track reaches `confirm_hits`. It then removes tracks unseen for
  `exit_after_s` (emitting `exit` if they had been confirmed), calls `_near_events`,
  appends all new events to the history and adds their salience to
  `pending_salience`. Returns the new events.
- **`_motion_event(t, tr)`** — compares the track's current distance with its
  distance about `motion_window_s` ago. If it changed by more than
  `motion_rel_change`, returns an `approach` or `recede` event and starts the
  cooldown. Approaching is given higher salience than receding.
- **`_near_events(t)`** — finds pairs of live, confirmed objects whose boxes are
  adjacent in the image and, when depth is available, within `near_depth_m` of each
  other in depth. The depth test is what stops a cup in the foreground from being
  "next to" a person standing behind it. Pairs of the same non-person class are
  skipped. A pair is reported once it has held for `near_hold_s`; pairs that were
  already adjacent when one of them first appeared are recorded silently.
- **`_where(tr)`** — "on the left, about 1.2 m away".
- **`active(t=None)`** — confirmed tracks, nearest first.
- **`describe(max_objects=8)`** — the instant, model-free description: counts per
  class, then one line per object with position and distance, then adjacency
  relations.
- **`recent_events(t, horizon_s=10)`** — events in the last few seconds.
- **`should_narrate(t, ...)`** — returns `True` when at least `min_gap_s` has passed
  since the last narration and either the accumulated event salience has reached the
  threshold or the scene has gone `idle_refresh_s` without a refresh. This is the
  event-driven trigger.
- **`mark_narrated(t)`** — resets the salience counter after a request is sent.
- **`grounding_prompt(t, question=None)`** — builds the text sent to the language
  model: the structured description, recent events, then either the user's question
  or an instruction to describe the scene in one sentence without inventing objects.

---

## `livescene/perception.py` — the vision models

- **`DEFAULT_VOCAB`** — the default list of object names given to the detector. Any
  noun phrase can be used; override with `--vocab`.
- **`OpenVocabTracker`**
  - `__init__(weights, vocab, device, imgsz, conf, tracker)` — loads YOLOE and sets
    the vocabulary.
  - `set_vocab(vocab)` — encodes the words with the MobileCLIP text encoder and
    installs the embeddings as the detector's classification weights. This is done
    once, so vocabulary size has no per-frame cost.
  - `__call__(frame_bgr)` — runs detection, segmentation and ByteTrack association in
    one call (`model.track(..., persist=True)`) and returns NumPy boxes, label
    strings, confidences, integer track IDs and mask polygons in frame pixels.
    Returns empty lists when nothing is tracked.
- **`MetricDepth`**
  - `__init__(model_id, device, short_side)` — loads Depth Anything V2 Metric Small.
  - `__call__(frame_bgr)` — resizes so the short side is about 266 px and both sides
    are multiples of the ViT patch size 14, converts BGR to normalised RGB on the
    GPU, runs the model and returns a low-resolution depth map in metres. The
    Hugging Face image processor is bypassed because its CPU preprocessing was a
    measurable part of the frame time.
- **`object_distance(depth, frame_hw, box, poly)`** — one distance per object. Scales
  the mask polygon to the depth map, rasterises it and takes the median depth inside.
  Falls back to the central half of the box when there is no usable mask.
- **`to_detections(...)`** — zips the tracker output and depth into `Detection`s.

## `livescene/narrator.py` — the language model

- **`Narrator.__init__`** — loads SmolVLM2-500M and its processor, turns off image
  tiling and caps the image at 512 px (64 visual tokens), and starts the worker
  thread.
- **`generate(frame_bgr, prompt)`** — one blocking call: builds a chat message with
  the image and prompt, generates up to `max_new_tokens` greedily, and decodes only
  the newly generated tokens.
- **`request(frame_bgr, prompt, t, question=None)`** — non-blocking. Stores the
  request in a single pending slot (replacing any request that has not started) and
  wakes the worker. A stale frame is never narrated after a newer one is queued.
- **`_loop()`** — the worker: waits for a request, generates, and publishes the
  result with its latency in `self.latest`. Errors are caught and reported as text so
  a failed generation cannot stop the camera loop.
- **`close()`** — stops the worker.

## `livescene/pipeline.py` — orchestration and drawing

- **`_color(track_id)`** — a stable colour per track ID.
- **`_label(img, text, org, color)`** — draws text on a dark plate.
- **`Pipeline.__init__`** — builds the tracker, optional depth model, optional
  narrator, the scene memory and a profiler.
- **`step(frame, t)`** — one frame through steps 2 to 6 above, each stage timed.
  Returns the new events.
- **`ask(frame, t, question)`** — sends a user question to the narrator with the
  grounding prompt.
- **`_collect_narration(t)`** — picks up a finished narration, records its latency
  and adds it to the log.
- **`draw(frame, fps)`** — mask overlays, boxes with name and distance for confirmed
  tracks, the last few events at the top left, and the latest narration wrapped along
  the bottom.
- **`close()`** — shuts the narrator down.

## `livescene/profiler.py` — timing

- **`Profiler.stage(name, sync=None)`** — context manager that times a block. When
  `sync` is `torch.cuda.synchronize`, it is called before both clock reads; without
  it, asynchronous GPU work would be charged to whichever later stage happens to wait
  for it.
- **`add(name, ms)`** — record an externally measured time.
- **`summary()`** — mean, median and 95th percentile per stage.
- **`last(name)`** — most recent sample.

## `run.py` — command line

Parses arguments, opens the source, builds the `Pipeline`, and runs the loop. Points
worth knowing:

- `CACHE` / `HF_HOME` at the top route all model downloads to
  `cv-envs\livescene`, and the working directory is switched to its `weights`
  subfolder because Ultralytics downloads into the current directory.
- For a file, time `t` is the video's own timeline (`frame / fps`), so the event log
  is identical between runs regardless of processing speed. For a webcam it is wall
  clock time.
- The first 10 frames are excluded from the end-to-end statistic (model warm-up).
- A background thread reads console lines as questions in live mode.
- At exit it waits for an in-flight narration, prints the latency table, and writes
  the JSON report if `--report` was given.

## `tests/test_memory.py`

Five tests on synthetic tracks, runnable without a GPU:

| Test | What it proves |
|------|----------------|
| `test_enter_needs_confirmation_and_exit_fires_once` | A 2-frame flicker is ignored; a real object gives exactly one enter and one exit |
| `test_label_flicker_is_voted_out` | A track that alternates cup / mug settles on the majority label |
| `test_approach_event_with_cooldown` | Walking from 4.0 m to 1.5 m gives an approach event, not a stream of them, and no recede |
| `test_near_respects_depth` | A cup sliding beside a person at the same depth is "next to"; the same image overlap 3 m behind is not |
| `test_narration_trigger_is_event_driven` | 15 s of an unchanging scene triggers exactly one narration |

## `tools/precision_check.py`

Runs the depth model in FP32 and under FP16 autocast on one frame, and prints both
latencies, the fraction of finite FP16 pixels and the relative difference. Use it on
any new GPU before deciding whether to enable half precision.
