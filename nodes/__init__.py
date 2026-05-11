import os
import sys
from pathlib import Path

# Add the nodes directory to sys.path so 'src' and other modules can be imported
NODES_DIR = Path(__file__).resolve().parent
if str(NODES_DIR) not in sys.path:
    sys.path.insert(0, str(NODES_DIR))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
