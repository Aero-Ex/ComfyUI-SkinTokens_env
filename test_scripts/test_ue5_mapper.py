import os
import sys
from pathlib import Path

# Add the project root to sys.path so we can import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.rig_package.parser.bpy import BpyParser
from src.rig_package.info.ue5_mapper import map_asset_to_ue5

def test_ue5_mapper(fbx_path):
    print(f"Loading FBX: {fbx_path}...")
    try:
        asset = BpyParser.load(filepath=fbx_path)
    except Exception as e:
        print(f"Failed to load FBX. Error: {e}")
        return

    joints = asset.joints
    parents = asset.parents
    original_names = asset.joint_names

    if joints is None or parents is None:
        print("Error: The loaded FBX does not contain joints or parents.")
        return

    print("\n--- Running UE5 Mapper ---")
    mapped_names = map_asset_to_ue5(joints, parents)

    print(f"{'Original Name':<25} | {'Mapped Name':<25} | {'Parent Index':<12}")
    print("-" * 70)
    
    matches = 0
    mismatches = 0
    
    # We only care about standard generated bones that our mapper handles
    for i in range(len(joints)):
        orig = original_names[i] if original_names else f"bone_{i}"
        mapped = mapped_names[i]
        parent_idx = parents[i]
        
        # Check if mapped name is a standard bone (not bone_X fallback)
        is_mapped_standard = not mapped.startswith("bone_")
        
        if is_mapped_standard:
            if orig.lower() == mapped.lower():
                matches += 1
                status = "OK"
            else:
                mismatches += 1
                status = "MISMATCH"
            print(f"{orig:<25} | {mapped:<25} | {status:<10}")
        else:
            print(f"{orig:<25} | {mapped:<25} | (skip/extra)")

    print("-" * 70)
    print(f"Analysis complete. Matches: {matches}, Mismatches: {mismatches}")

if __name__ == "__main__":
    import os
    cur_file = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(cur_file))
    comfy_root = os.path.dirname(os.path.dirname(repo_root))
    test_file = os.path.join(comfy_root, "input", "3d", "manny--unreal--engine-5", "source", "Manny_Unreal_Engine_5.fbx")
    test_ue5_mapper(test_file)
