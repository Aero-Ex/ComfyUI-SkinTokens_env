# Called by ComfyUI-Manager / comfy-cli after requirements.txt is installed.
# Builds this pack's isolated environment from nodes/comfy-env.toml (and pulls
# the prebuilt CUDA wheels declared there); installs nothing into the host.
from comfy_env import install

install()
