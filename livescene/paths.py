"""Where downloaded models and datasets are kept.

Import this before transformers or huggingface_hub so HF_HOME takes effect.
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Everything downloaded goes under <repo>/.cache unless LIVESCENE_CACHE points elsewhere.
CACHE = os.environ.get("LIVESCENE_CACHE", os.path.join(REPO, ".cache"))
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "hf"))
