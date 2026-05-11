"""Boot-time staging of tiny test fixtures (CPU-test support).

Copies bundled fixtures (see test-assets/) into ComfyUI's input directory so
test workflows can reference them by bare filename on any machine.

Idempotent and safe: never overwrites an existing file, and any failure is
swallowed -- a missing/unwritable fixture must never break pack import.
"""
import os
import shutil

FIXTURES = (
    "cube.obj",
)


def stage_test_fixtures():
    import folder_paths

    try:
        input_dir = folder_paths.get_input_directory()
    except Exception:
        return
    try:
        os.makedirs(input_dir, exist_ok=True)
    except OSError:
        return

    pack_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(pack_dir, "test-assets")
    for name in FIXTURES:
        src = os.path.join(assets_dir, name)
        dst = os.path.join(input_dir, name)
        try:
            if os.path.exists(dst) or not os.path.isfile(src):
                continue
            shutil.copyfile(src, dst)
            print(f"[SkinTokens] Staged test fixture: {name} -> input dir")
        except OSError:
            continue
