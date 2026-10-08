"""Asynchronous language worker.

The vision-language model is far slower than the camera, so language jobs run on their
own thread and the perception loop never waits for them. There is one pending slot: a
newer job replaces an older one that has not started, except that a user's question is
never displaced by an automatic description.
"""

import threading
import time


class Narrator:
    def __init__(self):
        self.latest = None  # dict(text, t_request, latency_ms, question, ...extras from the job)
        self.busy = False
        self._answering = False
        self._pending = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def question_waiting(self):
        """True while a user's question is queued or being answered."""
        p = self._pending
        return self._answering or (p is not None and p[2] is not None)

    def request(self, job, t, question=None):
        """Queue `job`, a callable returning a dict with at least `text`. Returns False if
        it was refused because a question is already waiting."""
        with self._cv:
            if question is None and self._pending is not None and self._pending[2] is not None:
                return False
            self._pending = (job, t, question)
            self._cv.notify()
            return True

    def _loop(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                (job, t, question), self._pending = self._pending, None
                self.busy = True
                self._answering = question is not None
            t0 = time.perf_counter()
            try:
                result = dict(job())
            except Exception as e:  # keep the camera loop alive if a job fails
                result = {"text": f"[language model error: {type(e).__name__}: {e}]", "error": True}
            result.update(t_request=t, question=question, latency_ms=(time.perf_counter() - t0) * 1000.0)
            self.latest = result
            self._answering = False
            self.busy = False

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
