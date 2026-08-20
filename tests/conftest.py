"""
classify.py now loads its embedding model in a background thread (so
app.py's server can open its port immediately instead of blocking on a
multi-second download+embed step -- see classify.py for why). Tests that
assert specific semantic-classifier behavior need the model to have
actually finished loading first, or they'd race against that thread and
become flaky. This runs once, automatically, before any test.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import classify

classify.wait_for_model_ready(timeout=60)
