import bpy
import sys
import os

def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cur_file = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(cur_file))
    comfy_root = os.path.dirname(os.path.dirname(repo_root))
    
    fbx_path = os.path.join(comfy_root, "input", "3d", "manny--unreal--engine-5", "source", "Manny_Unreal_Engine_5.fbx")
    try:
        bpy.ops.import_scene.fbx(filepath=fbx_path)
    except Exception as e:
        print(f"Failed to import FBX: {e}")
        sys.exit(1)
        
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    if not armatures:
        print("No armature found in the imported FBX scene.")
        sys.exit(1)
        
    armature = armatures[0]
    out_file = os.path.join(repo_root, "test_scripts", "manny_bones.txt")
    
    with open(out_file, "w") as f:
        f.write(f"Armature name: {armature.name}\n")
        f.write("--- BONES HIERARCHY ---\n")
        for bone in armature.pose.bones:
            parent_name = bone.parent.name if bone.parent else "None"
            head_pos = list(bone.head)
            f.write(f"BONE: '{bone.name}' | PARENT: '{parent_name}' | HEAD: {head_pos}\n")
            
    print(f"Bones dumped to {out_file}")

if __name__ == "__main__":
    main()
