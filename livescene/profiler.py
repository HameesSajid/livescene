"""Per-stage latency bookkeeping for the real-time loop."""

import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np


class Profiler:
    """Keeps a sliding window of wall-clock timings (ms) per named stage."""

    def __init__(self, window=300):
        self.samples = defaultdict(lambda: deque(maxlen=window))

    @contextmanager
    def stage(self, name, sync=None):
        """Time a block. `sync` is an optional callable (e.g. torch.cuda.synchronize)
        run before each clock read so queued GPU work is charged to the right stage."""
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync is not None:
                sync()
            self.samples[name].append((time.perf_counter() - t0) * 1000.0)

    def add(self, name, ms):
        self.samples[name].append(float(ms))

    def summary(self):
        out = {}
        for name, vals in self.samples.items():
            a = np.asarray(vals, dtype=np.float64)
            out[name] = {
                "n": int(a.size),
                "mean_ms": float(a.mean()),
                "p50_ms": float(np.percentile(a, 50)),
                "p95_ms": float(np.percentile(a, 95)),
            }
        return out

    def last(self, name, default=0.0):
        vals = self.samples.get(name)
        return vals[-1] if vals else default
