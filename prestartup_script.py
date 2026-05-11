# Runs before the ComfyUI server boots. Installs nothing -- reports environment
# health and applies the platform hygiene comfy-env needs.
from comfy_env import setup_env

setup_env()
