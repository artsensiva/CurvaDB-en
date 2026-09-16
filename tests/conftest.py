"""Package isn't installed (not even editable) -- add src/ to sys.path
once for all tests, instead of duplicating sys.path.insert in every file."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
