# Code guide

A file-by-file, function-by-function explanation of LiveScene. Read
[CONCEPTS.md](CONCEPTS.md) first for the ideas; this document is about the code.

## The two loops

**Fast loop (every frame, main thread).**

1. `run.py` reads a frame and resizes it to the profile's width.
2. `OpenVocabDetector.detect` returns boxes, classes, confidences and masks for the
   current vocabulary.
3. `Tracker.update` assigns a persistent id to each detection; `observe` packs the
   result into `Observation` records.
4. On every third frame, `MetricDepth` produces a depth map in metres.
5. `to_detections` computes one distance per object (median depth inside its mask).
6. `SceneMemory.update` merges the detections into its tracks and returns events, but
   only for objects whose label is *trusted* (see below).
7. `Pipeline.draw` renders masks, boxes, the event ticker, the description, the latest
   answer and the question box.

**Slow loop (language thread, when something changed).**

1. `Pipeline._maybe_describe` decides a description is due: at start, after an event,
   when the detector keeps reporting a label nothing has confirmed, or every 15 s.
2. `propose_verify_describe` asks the language model what it sees and for a description,
   checks each object with a second detector, asks the model again about anything the
   detector could not find, and removes sentences about objects that fail both checks.
3. Back on the main thread, `Pipeline.poll` shows the description, marks the confirmed
   labels as trusted, and adds them to the tracking detector's vocabulary.

A label therefore produces events only when two independent models have reported it:
the detector (it is being tracked) and the language model (it named it).

---

## `livescene/config.py` — profiles

- **`Config`** — one frozen dataclass per profile: detector and verifier weights, depth
  model and its input size, language model spec, image side and frame count given to
  the language model, tokens per description, whether to rewrite or only cut sentences,
  detector input size, frame width, the verification threshold, and `two_step`.
- **`PROFILES`** — `laptop` (6 GB), `colab-t4` (12 to 16 GB), `colab` (22 GB and up).
- **`pick(name, **overrides)`** — returns a profile; `auto` chooses by GPU memory.
  Overrides that are `None` are ignored, so command-line flags can be passed straight in.

## `livescene/paths.py` — where downloads go

Defines `REPO` (the project folder) and `CACHE` (`<project>/.cache`, or the
`LIVESCENE_CACHE` environment variable if set), and points Hugging Face's `HF_HOME` at
`CACHE/hf`. Every script imports it before the model libraries, so all weights and
datasets land in one git-ignored folder inside the project on any machine.

## `livescene/vlm.py` — language model backends

Every backend has the same call, `generate(frames_bgr, prompt, max_new_tokens) -> str`,
where `frames_bgr` is a list of frames (one for a still, several for a short clip).

- **`_to_pil(frame_bgr, max_side)`** — BGR array to an RGB PIL image, shrunk to
  `max_side`. Capping the size is what bounds the number of visual tokens.
- **`HFVLM`** — any Hugging Face image-text-to-text chat model, run in-process.
  - `__init__(model_id, device, dtype, max_side, load_4bit)` — loads processor and model.
    For SmolVLM-style processors it turns off image tiling. `load_4bit` loads with
    bitsandbytes NF4 quantisation (needs the `bitsandbytes` package).
  - `generate(...)` — builds one chat turn with all frames then the prompt, generates
    greedily, and decodes only the new tokens.
- **`ServerVLM`** — a model behind an OpenAI-compatible HTTP endpoint (for example
  llama.cpp's server). Frames are sent as base64 JPEGs.
- **`load_vlm(spec)`** — `spec` is a model id, optionally with `@fp16` or `@4bit`, or
  `server:<url>`.

## `livescene/describe.py` — propose, verify, describe

Plain text handling, unit-tested without a model.

- **Prompts** — `PROPOSE_PROMPT` (objects and description in one reply, for capable
  models), `DESCRIBE_PROMPT` and `LIST_PROMPT` (the same two things asked separately,
  for small models), `REWRITE_PROMPT`, `QA_PROMPT`.
- **`_singular(word)`**, **`clean_object(name)`** — normalise a proposed object:
  lower-case, drop articles and counts, singular last word, at most three words, and
  every word for a human becomes `person` so it matches the tracker's label.
- **`parse_proposal(text)`** — splits an `OBJECTS: ... DESCRIPTION: ...` reply into a
  list of objects and the description, tolerating bullets, new lines and missing fields.
- **`parse_list(text)`** — objects from a free-form list reply.
- **`mentions(text, term)`** — whole-word match allowing a simple plural; `person`
  matches any word for a human.
- **`split_sentences`**, **`strip_rejected(description, rejected)`** — remove every
  sentence that names a rejected object.
- **`verify(objects, evidence, threshold)`** — splits objects into found and missed by
  detector confidence. Names sharing a head noun ("table", "dining table") share
  evidence, because they compete for the same region inside the detector.
- **`self_check(vlm, frames, name)`** — asks the model "Is there a ... in the image?"
- **`qa_prompt(question, facts)`** — the prompt for a user's question, with measured
  positions and distances of trusted objects when there are any.
- **`propose_verify_describe(...)`** — the whole loop. Returns `description`, `accepted`
  (found by the detector), `kept` (missed by the detector but the model stood by it),
  `rejected`, the detector `evidence`, and `strict` (the text if everything the detector
  missed had been cut, used to compare the two policies on identical captions).

## `livescene/perception.py` — the vision models

- **`DEFAULT_VOCAB`** — a broad starting vocabulary. Labels that come only from this
  list are treated as unconfirmed guesses.
- **`Observation`** — one tracked object in one frame: id, label, confidence, box, mask.
- **`OpenVocabDetector`**
  - `embed(names)` — text embeddings for words, cached per word. Encoding a new word
    takes about a second, so the live pipeline does it on the language thread.
  - `adopt(embeddings)` — take embeddings computed by another detector with the same
    weights.
  - `set_vocab(vocab)` — install a vocabulary. With cached embeddings this takes about
    a millisecond, which is what makes a changing vocabulary practical.
  - `detect(frame, conf, masks)` — one forward pass; masks are returned at frame
    resolution.
  - `check(frame, names, conf, competitors)` — verification query: prompt with `names`
    and return the best confidence for each.
- **`Tracker`** — ByteTrack owned by this project. Ultralytics' built-in tracker is
  thrown away whenever the classes change, which would reset every id. Thresholds: a
  detection of 0.3 or more drives matching, 0.1 to 0.3 may only keep an existing track
  alive, and a new id needs 0.4. Lost tracks are kept for 3 seconds.
- **`observe(result, pairs)`** — detector result plus tracker assignments to
  `Observation`s.
- **`MetricDepth`** — Depth Anything V2 metric model; preprocessing is done on the GPU.
- **`object_distance(depth, frame_hw, box, mask)`** — median depth inside the mask,
  falling back to the centre of the box.
- **`to_detections(observations, depth, frame_hw)`** — wraps observations for the memory.

## `livescene/memory.py` — the scene memory

Pure Python and NumPy, no model.

- **`Detection`**, **`Event`**, **`Track`** — data records. A track is `confirmed` once
  seen `confirm_hits` times and `announced` once it is also trusted; only announced
  tracks create events or appear in descriptions. Its label is a confidence-weighted
  vote over the last 30 frames, so it follows a vocabulary change within a second or two.
- **`_box_gap`**, **`_iou`**, **`_plural`** — small helpers.
- **`SceneMemory`**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `confirm_hits` | 3 | Frames before a detection counts as real |
| `exit_after_s` | 1.0 | Seconds unseen before an object is gone |
| `motion_window_s`, `motion_rel_change`, `motion_cooldown_s` | 1.5, 0.25, 3.0 | Approach / recede detection |
| `near_gap_frac`, `near_depth_m`, `near_hold_s`, `near_release_s` | 0.03, 0.6, 0.5, 1.5 | "Next to" detection with hysteresis |
| `require_trust` | False | Only labels confirmed by a second source create events |
| `trust_ttl_s` | 90 | How long a confirmation lasts |
| `relink_iou` | 0.3 | Overlap needed to treat a new tracker id as an existing object |

  - `trust(labels, t)`, `is_trusted(label, t)` — record and query confirmations.
  - `resolve(tracker_id)` — the id the memory uses for a tracker id (they differ after
    a re-link).
  - `_relink(d, t, present)` — if a new tracker id lands on a same-label track that just
    went missing, it is that track; no "left" and "appeared" pair is produced.
  - `update(t, detections)` — the per-frame entry point: update or create tracks, vote
    labels, smooth distance, announce, detect motion, expire, detect adjacency.
  - `_motion_event`, `_near_events` — approach / recede and "next to" events.
  - `active()`, `describe()`, `facts()` — trusted objects as a list, as a structured
    text summary, and as measurement lines for question answering.
  - `candidates(t)` — labels the detector keeps reporting that nothing has confirmed;
    the pipeline uses them as a reason to ask the language model for a fresh look.
  - `should_narrate`, `mark_narrated`, `grounding_prompt` — the event trigger, and the
    first version's prompt, kept so the benchmark can still measure it.

## `livescene/narrator.py` — the language thread

- **`Narrator.request(job, t, question)`** — queue a callable that returns a dict with
  `text`. One pending slot; a description never displaces a waiting question.
- **`question_waiting`** — true while a question is queued or being answered.
- **`_loop`** — runs jobs, records latency, and turns an exception into a visible
  message so a failed generation cannot stop the camera loop.

## `livescene/pipeline.py` — orchestration and drawing

- **`step(frame, t)`** — the fast loop above, with each stage timed.
- **`_frames_for_vlm(frame)`** — the current frame, preceded by earlier frames sampled
  from the last few seconds when the profile uses more than one.
- **`_maybe_describe(frame, t)`** — decides whether to start the slow loop and builds
  its context (how many frames, recent tracker events).
- **`_describe_job(frames, context)`** — runs on the language thread; also prepares the
  text embeddings of newly accepted objects so the main thread never stalls on them.
- **`ask(frame, t, question)`** — queue a user's question with the measured facts.
- **`poll(t)`** — collect a finished job: show it, log it, trust accepted labels, update
  the vocabulary.
- **`_update_vocab`** — vocabulary = starting list + confirmed additions (at most 30,
  dropped when not re-confirmed for 90 s).
- **`draw(frame, fps, typing, paused)`** — confirmed-and-trusted objects get a coloured
  mask and a label with distance; unconfirmed detector guesses get a thin grey box and
  a question mark. Panels at the bottom: the question box while typing, the
  description, and the latest question and answer.

## `livescene/profiler.py` — timing

- **`Profiler.stage(name, sync)`** — times a block; with `torch.cuda.synchronize` as
  `sync`, GPU work is charged to the stage that issued it.
- **`add`**, **`summary`**, **`last`** — record a time, per-stage mean / median / p95,
  most recent sample.

## `run.py` — command line and window

- Resolves the profile, opens the source, builds the `Pipeline`.
- For a file, time `t` is the video's own timeline, so the log is reproducible; for a
  webcam it is wall-clock time.
- **Keys:** `a` opens a question box in the window (the video pauses while you type,
  Enter sends, Esc cancels), `space` pauses, `d` forces a description, `q` quits.
  Up to 12 queued keystrokes are read per frame so typing keeps up at low frame rates.
- `--ask "..."` asks a question without a window, at `--ask-at` seconds.
- At exit it waits for unfinished language jobs, prints the latency table and writes the
  JSON report.

## `tools/eval_hallucination.py` — hallucination benchmarks

- **`build_pope`** — POPE probes from COCO annotations: 3 present objects and 3 absent
  ones per image for each of the random, popular and adversarial settings.
- **`question`**, **`parse_yes_no`**, **`wilson`**, **`score`** — question text, answer
  parsing (POPE's rule), a 95 % confidence interval, and accuracy / precision / recall /
  F1 / yes-ratio.
- **`detector_memory`**, **`load_detector`** — helpers for the modes that use a detector.
- **`run_pope`** — asks every probe in `plain` or `grounded` mode.
- **`run_chair`** — captions every image in `plain`, `grounded` or `verified` mode and
  scores the captions; in `verified` mode it also scores the strict policy.
- Per-item logs go to `.cache/eval/*.jsonl`; summaries to `--out`.

## `tools/chair.py` — the CHAIR scorer

A dependency-free re-implementation of the official caption-to-object matching:
synonym list, two-word objects, "baby elephant" is an elephant, plural handling.
`Chair.objects(caption)` lists the COCO categories a caption mentions;
`Chair.score(captions, truths)` returns CHAIR_i, CHAIR_s, recall and average length.

## `tools/eval_coco.py` — detection accuracy

Standard COCO box mAP on a fixed val2017 subset. YOLOE models are prompted with the 80
class names; other weights use their own classes. `fetch_annotations` and
`fetch_images` download what is missing into `.cache/data/coco`.

## `tools/precision_check.py`

Runs the depth model in FP32 and under FP16 autocast and reports both latencies and
their numerical difference, to decide whether half precision is worth using on a GPU.

## Tests

| File | Covers |
|------|--------|
| `tests/test_memory.py` | Confirmation, exit, label voting, approach, depth-aware adjacency, the event-driven trigger, trust gating, trust expiry, re-linking after an id switch, labels following a vocabulary change |
| `tests/test_describe.py` | Parsing well-formed and messy replies, object normalisation, plural and word-boundary matching, sentence removal, noun-family evidence, the strict and lenient policies, the rewrite check, the question prompt |
| `tests/test_chair.py` | Synonyms and plurals, two-word objects, baby animals, the three scores |

All three run without a GPU: `python tests/test_memory.py` and so on.
