import os
import folder_paths

# Add custom model folder for SkinTokens on the host
skintoken_models_dir = os.path.join(folder_paths.models_dir, "skintoken")
if not os.path.exists(skintoken_models_dir):
    os.makedirs(skintoken_models_dir, exist_ok=True)

# Register the skintoken model type
folder_paths.folder_names_and_paths["skintoken"] = (
    [skintoken_models_dir],
    folder_paths.supported_pt_extensions,
)

from comfy_env import register_nodes

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()

# Stage tiny CPU-test fixtures (e.g. cube.obj) into the input dir so test
# workflows resolve them by bare filename. Guarded: staging must never fail
# pack import (see prestartup.py).
try:
    from .prestartup import stage_test_fixtures
    stage_test_fixtures()
except Exception:
    pass

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
