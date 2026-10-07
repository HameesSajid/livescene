# Concepts: what LiveScene uses, what it could use, and why the chosen set is enough

LiveScene has one goal: describe a live camera feed in language, in real time, on a
laptop GPU (GTX 1660 Ti, 6 GB), and be right about *what* is there, *where* it is, and
*what changed*. Every concept below is judged against that goal.

## 1. Concepts used

| # | Concept | Model / method | What it contributes |
|---|---------|----------------|---------------------|
| 1 | Open-vocabulary detection | YOLOE, text-prompted | Finds objects named by free text, not a fixed 80-class list |
| 2 | Vision-language alignment | CLIP-style text encoder (MobileCLIP) inside YOLOE | Turns the prompt words into the classifier weights |
| 3 | Instance segmentation | YOLOE mask head | Pixel-accurate object regions, used for depth sampling and display |
| 4 | Multi-object tracking | ByteTrack (Kalman filter + two-stage IoU association) | Stable identity per object across frames |
| 5 | Monocular metric depth | Depth Anything V2 Small (DINOv2 ViT-S backbone + DPT head) | Distance in metres from a single RGB camera |
| 6 | Self-supervised foundation features | DINOv2 (inside the depth model) | The reason depth generalises to unseen scenes |
| 7 | Vision-language model | SmolVLM2-500M (SigLIP vision encoder + small LLM) | Natural-language description and question answering |
| 8 | Grounded generation | Scene-memory text injected into the VLM prompt | Cuts hallucination: the VLM is told what the detector, tracker and depth measured |
| 9 | Spatio-temporal scene memory | `SceneMemory` (this repo) | Object state over time, spatial relations, discrete events |
| 10 | Event-driven inference | `should_narrate` (this repo) | The slow model runs only when something changed |
| 11 | Asynchronous multi-rate scheduling | Narrator thread, depth every N frames | Fast models run every frame, slow ones at their own rate |
| 12 | Systems measurement | `Profiler`, synchronised GPU timing | Per-stage mean / p50 / p95 latency and peak VRAM |

### How they fit together

```
camera frame
   |
   +--> YOLOE (text prompts) --> boxes + masks --> ByteTrack --> track IDs
   |                                                              |
   +--> Depth Anything V2 (every N frames) --> metres ----------->+
                                                                  v
                                                           SceneMemory
                                              state, relations, events, text
                                                   |                |
                               instant structured description       | only when an event fires
                                                                    v
                                         SmolVLM2 (own thread): frame + memory text
                                                                    |
                                                     narration / answer to a question
```

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

**Language: SmolVLM2-500M instead of BLIP (the earlier project), LLaVA, Qwen2-VL-7B,
or a cloud API.**
BLIP is a captioner: it cannot answer questions or take extra context. 7B-class VLMs
do not fit in 6 GB next to the other models without heavy quantisation, and would take
several seconds per answer. A cloud API would be more fluent but adds network latency,
cost and a privacy problem for a live camera. A 500M-parameter VLM fits, answers in
about a second, and its weaker reasoning is compensated by grounding (concept 8).

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
| Larger VLM with 4-bit quantisation | Better reasoning | Trades away the latency budget |
| Audio output (TTS) | An assistive, hands-free product | Product feature rather than a vision concept |

## 4. Why the chosen set is sufficient

A description of a scene needs to answer four questions, and each one is covered by
exactly one component with no gap and no duplicate:

1. **What is there?** Open-vocabulary detection with masks (concepts 1 to 3).
2. **Which one is which, over time?** Tracking (4).
3. **Where is it in space?** Metric depth plus image position (5, 6).
4. **What is happening, in words?** Memory, events and a grounded VLM (7 to 10).

Concepts 11 and 12 are what make the first four run together in real time and prove
that they do. Each item in section 3 either sharpens one of these answers at a cost
the hardware cannot pay in real time, or answers a fifth question (reading text,
building a map, naming fine-grained actions) that is outside the stated goal. Adding
them would make the system heavier without making its core claim more true, which is
why they are listed as extensions and not built in.

The measured numbers that back the real-time claim are in the README.
