# Concepts: what LiveScene uses, what it could use, and why the chosen set is enough

LiveScene has one goal: describe a camera feed in language and be right about *what* is
there, *where* it is, and *what changed*, without inventing things. It runs live on a
laptop GPU (GTX 1660 Ti, 6 GB) with small models and on a large Colab GPU with big ones.
Every concept below is judged against that goal.

## 1. Concepts used

| # | Concept | Model / method | What it contributes |
|---|---------|----------------|---------------------|
| 1 | Open-vocabulary detection | YOLOE, text-prompted | Finds objects named by free text, not a fixed 80-class list |
| 2 | Vision-language alignment | CLIP-style text encoder (MobileCLIP) inside YOLOE | Turns the prompt words into the classifier weights |
| 3 | Instance segmentation | YOLOE mask head | Pixel-accurate object regions, used for depth sampling and display |
| 4 | Multi-object tracking | ByteTrack (Kalman filter + two-stage IoU association) | Stable identity per object across frames |
| 5 | Monocular metric depth | Depth Anything V2 Small (DINOv2 ViT-S backbone + DPT head) | Distance in metres from a single RGB camera |
| 6 | Self-supervised foundation features | DINOv2 (inside the depth model) | The reason depth generalises to unseen scenes |
| 7 | Vision-language model | SmolVLM2-500M on the laptop, Qwen2.5-VL-3B / 7B on Colab | Natural-language description and question answering |
| 8 | Cross-model verification | Propose, verify, describe (`describe.py`) | An object is described only if the language model names it and the detector finds it |
| 9 | Dynamic vocabulary | Confirmed objects become the detector's text prompts at run time | The tracker follows what is in this scene instead of a fixed word list |
| 10 | Spatio-temporal scene memory with trust | `SceneMemory` (this repo) | Object state over time, relations, and events only for labels both models reported |
| 11 | Event-driven inference | `Pipeline._maybe_describe` | The slow model runs only when something changed |
| 12 | Asynchronous multi-rate scheduling | Language thread, depth every N frames | Fast models run every frame, slow ones at their own rate |
| 13 | Hallucination measurement | POPE and CHAIR on COCO (`tools/eval_hallucination.py`) | "Accurate" is a number, measured before and after each design change |
| 14 | Systems measurement | `Profiler`, synchronised GPU timing | Per-stage mean / p50 / p95 latency and peak VRAM |
| 15 | Hardware profiles | `config.py` | One code base from a 6 GB laptop to a 40 GB Colab GPU |

### How they fit together

```
FAST LOOP, every frame
camera frame --> YOLOE (current vocabulary) --> boxes + masks --> ByteTrack --> track ids
             --> Depth Anything V2 (every Nth frame) --> metres
                                   |
                                   v
                    SceneMemory: tracks, distances, relations
                    events only for TRUSTED labels  ----------------> overlay, event log

SLOW LOOP, when something changed (own thread)
frame(s) --> language model: "what objects do you see? describe the scene"
                 |  proposed objects + description
                 v
             second YOLOE prompted with exactly those objects
                 |  found / not found
                 v
             remove sentences about objects that were not found  --> description
                 |
                 +--> found objects become TRUSTED and join the fast loop's vocabulary
```

### The design decision that mattered most

The first version passed the detector's labels to the language model as facts
("Tracked objects: truck #3 ..."). It seemed obviously helpful. Measured on 100 COCO
images with CHAIR, it was harmful:

| How the description is produced | Captions with an invented object |
|---------------------------------|----------------------------------|
| Detector labels given to the language model as facts (first version) | 31 % |
| Language model alone | 20 % |
| Language model proposes, detector must confirm (current) | 5 % |

A zero-shot detector forced to choose from a fixed word list assigns the nearest word
to everything it sees, so a hand cart becomes a "truck". A small language model told
that a truck is present will say so. Reversing the direction of trust fixes both: the
language model names things freely from the pixels, and the detector, prompted with
exactly those names, only has to answer whether each one is there. Neither model's
guess is accepted on its own.
## 2. Why each choice, and what was rejected

**Detection: YOLOE instead of YOLOv8 (the earlier project), Grounding DINO, or OWLv2.**
YOLOv8 can only output the 80 COCO classes it was trained on. Grounding DINO and OWLv2
are open-vocabulary and more accurate on rare categories, but run at a few frames per
second on this GPU because they fuse text and image with cross-attention on every
frame. YOLOE re-parameterises the text embeddings into the detection head once, when
the vocabulary is set, so the per-frame cost is the same as a closed-set YOLO. It also
outputs masks, which removes the need for a separate segmenter. The price is measured,
not hidden: on 1,000 COCO val images YOLOv8m scores 0.526 mAP on its 80 trained
classes, while zero-shot YOLOE-26s scores 0.383 and YOLOE-26m 0.438. A description
system that can only ever name 80 things is the larger limitation, so the trade is
taken, and the medium model is available when accuracy matters more than speed.

**Segmentation: YOLOE masks instead of SAM 2.**
SAM 2 gives sharper masks and its own video memory, but even the tiny variant costs
more per frame than the whole YOLOE pass, and the masks here are only used to pick the
depth pixels that belong to an object and to draw overlays. A median over a slightly
loose mask gives the same distance as a median over a perfect one.

**Tracking: ByteTrack instead of BoT-SORT with ReID, or DeepSORT.**
Appearance-based ReID helps when objects are occluded for a long time or the camera
moves fast. For a mostly static camera at 20+ FPS, motion plus IoU association is
enough, and it adds under a millisecond. The memory layer also votes on the label over
the life of a track, which absorbs most open-vocabulary label flicker.

**Depth: Depth Anything V2 Small (metric) instead of stereo, a depth sensor, relative
depth, or the Large model.**
Stereo and RGB-D need extra hardware. Relative depth cannot say "1.2 m away" or test
whether two overlapping objects are really next to each other. The Large model is more
accurate but about ten times slower. The Small metric model gives usable metres at
low resolution in real time.

**Language: SmolVLM2-500M on the laptop and Qwen2.5-VL on Colab, instead of BLIP (the
earlier project) or a cloud API.**
BLIP is a captioner: it cannot answer questions or follow a prompt format. A cloud API
would be more fluent but adds network latency, cost and a privacy problem for a live
camera. On 6 GB the choice is forced: a 500M-parameter model is what fits next to the
detector and depth model in full precision and answers a question in about a second.
Its weakness is measured, not hidden (73 % accuracy on adversarial yes/no probes), and
verification is what compensates for it. With more memory the same code loads
Qwen2.5-VL-3B or 7B, which read several frames per call and follow the combined
"objects, then description" format that the small model cannot.
**Verification: strict instead of lenient, and why not a decoding-time method.**
Two policies were measured for an object the detector cannot find. Strict cuts it.
Lenient first asks the language model a direct yes/no question and cuts only if it
then says no. Lenient left 17 % of captions with an invented object and strict 5 %,
because the small model stands by almost everything it has said. Strict costs about
4 points of recall, since a detector miss on a real object also removes a true
sentence; that trade is taken, and the profile can switch it. Decoding-time methods
that change how tokens are sampled (contrastive decoding and similar) need two forward
passes per token or access to attention maps, and they cannot use the one extra source
of evidence this system already has running, the detector.

**Tracking: an own ByteTrack instance instead of the detector library's built-in one.**
The built-in tracker is discarded whenever the detector's classes change, which would
reset every identity each time the vocabulary is updated. Owning the tracker keeps ids
stable across vocabulary changes and allows two extra safeguards in the memory: a new
identity needs a confident detection, and a new id that lands on a just-lost object of
the same label is re-linked to it. Together with the trust rule this cut the events on
the two sample clips from 49 to 16 and from 88 to 28.

**Scheduling: event-driven instead of captioning every frame or every N seconds.**
Per-frame captioning is impossible at these speeds and would repeat itself. Fixed
interval captioning wastes calls on static scenes and is late on fast changes. Firing
on events ties language cost to how much the scene changes.

## 3. Concepts that could be added, and when they would be worth it

| Concept | What it would add | Why it is not in the core |
|---------|-------------------|---------------------------|
| TensorRT / ONNX export, INT8 quantisation | 1.5 to 3x faster detector and depth | Worth doing once accuracy is fixed; changes deployment, not capability |
| FP16 inference | Roughly 2x on GPUs with tensor cores | Measured slower than FP32 on the GTX 1660 Ti (86 ms vs 29 ms for depth), so this build stays FP32 |
| SAM 2 video propagation | Sharper masks, tracking through occlusion | Too slow on 6 GB; right choice on a Colab or RTX-class GPU |
| ReID embeddings (BoT-SORT, OSNet) | Identity kept across long occlusions and re-entry | Needed for multi-camera or crowded scenes |
| Pose / hand keypoints | Actions such as "picks up", "waves" | Next most valuable addition for richer events |
| Video action recognition (VideoMAE, V-JEPA) | Activity labels over clips | Overlaps with what the VLM already describes |
| 3D lifting: camera intrinsics, point clouds, Gaussian splatting, SLAM | A persistent 3D map | A different project (mapping), not description |
| OCR | Reading signs and screens | Easy add-on; scene text is a niche case here |
| Fine-tuning (LoRA on the VLM, detector distillation) | Domain-specific accuracy | Needs a target domain and labelled data to be meaningful |
| A 2B to 3B language model on the laptop (4-bit) | Better descriptions and answers locally | Needs one more package and a 7.5 GB download; slower per sentence |
| Audio output (TTS) | An assistive, hands-free product | Product feature rather than a vision concept |

## 4. Why the chosen set is sufficient

A description of a scene needs to answer four questions, and each one is covered by
exactly one component with no gap and no duplicate:

1. **What is there?** Open-vocabulary detection with masks (concepts 1 to 3), with the
   vocabulary supplied by the language model (9).
2. **Which one is which, over time?** Tracking (4).
3. **Where is it in space?** Metric depth plus image position (5, 6).
4. **What is happening, in words?** A vision-language model (7).
5. **Is that actually true?** Cross-model verification and trust (8, 10), measured
   with standard hallucination benchmarks (13).

Concepts 11, 12, 14 and 15 are what make these run together at interactive speed on
small hardware, scale to large hardware, and prove what the system does. Each item in section 3 either sharpens one of these answers at a cost
the hardware cannot pay in real time, or answers a different question (reading text,
building a map, naming fine-grained actions) that is outside the stated goal. The one
known gap inside the goal is that only objects are verified, not attributes or actions;
that is listed in the README as a limitation and is the next thing to build. Adding
them would make the system heavier without making its core claim more true, which is
why they are listed as extensions and not built in.

The measured numbers that back the real-time claim are in the README.
