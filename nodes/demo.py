import argparse
import atexit
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Iterable, Optional, Tuple

import gradio as gr
import requests
from torch import Tensor
from tqdm import tqdm



os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"
gr.TEMP_DIR = "tmp_gradio"

from src.data.dataset import DatasetConfig, RigDatasetModule
from src.data.transform import Transform
from src.model.tokenrig import TokenRigResult
from src.tokenizer.parse import get_tokenizer
from src.server.spec import (
    BPY_SERVER,
    get_model,
    object_to_bytes,
    bytes_to_object,
)
from src.data.vertex_group import voxel_skin

def get_default_ckpt():
    import os
    try:
        import folder_paths
        skintoken_models_dir = os.path.join(folder_paths.models_dir, "skintoken")
    except ImportError:
        cur_file = os.path.abspath(__file__)
        comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(cur_file)))
        skintoken_models_dir = os.path.join(comfy_root, "models", "skintoken")
    return os.path.join(skintoken_models_dir, "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt")

MODEL_CKPTS = [
    get_default_ckpt(),
]

HF_PATHS = [
    "None",
]


class BoneNode:
    def __init__(self, fbx_node, parent=None):
        self.fbx_node = fbx_node
        self.name = fbx_node.GetName()
        self.parent = parent
        self.children = []
        
        # Local transforms
        self.t = fbx_node.LclTranslation.Get()
        self.r = fbx_node.LclRotation.Get()
        self.s = fbx_node.LclScaling.Get()
        
    def add_child(self, child_node):
        self.children.append(child_node)

def build_tree(fbx_node, parent_bone=None):
    bone = BoneNode(fbx_node, parent_bone)
    for i in range(fbx_node.GetChildCount()):
        child_bone = build_tree(fbx_node.GetChild(i), bone)
        bone.add_child(child_bone)
    return bone

def is_skeleton(bone):
    node_attr = bone.fbx_node.GetNodeAttribute()
    return node_attr and node_attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton

def get_skeleton_joints(bone, joints_list):
    if is_skeleton(bone):
        joints_list.append(bone)
    for child in bone.children:
        get_skeleton_joints(child, joints_list)

def rename_joints_in_blender(input_file: str, output_file: str, format_type: str, convention: str):
    import tempfile
    
    script_content = """import sys
import bpy
from mathutils import Matrix

def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) < 4:
        sys.exit(1)
    input_file, output_file, format_type, convention = args[0], args[1], args[2], args[3]
    
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if format_type == "glb":
        bpy.ops.import_scene.gltf(filepath=input_file)
    elif format_type == "fbx":
        bpy.ops.import_scene.fbx(filepath=input_file)
        
    armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
    if not armatures:
        sys.exit(1)
    armature = armatures[0]
    
    class BoneNode:
        def __init__(self, bpy_bone, parent=None):
            self.bpy_bone = bpy_bone
            self.name = bpy_bone.name
            self.parent = parent
            self.children = []
            parent_matrix = bpy_bone.parent.matrix_local if bpy_bone.parent else Matrix.Identity(4)
            local_matrix = parent_matrix.inverted() @ bpy_bone.matrix_local
            self.t = local_matrix.to_translation()
            self.head = bpy_bone.head

    def build_tree(bpy_bone, parent_node=None):
        node = BoneNode(bpy_bone, parent_node)
        for child in bpy_bone.children:
            child_node = build_tree(child, node)
            node.children.append(child_node)
        return node

    roots = [b for b in armature.data.bones if b.parent is None]
    if not roots:
        sys.exit(1)
        
    root_nodes = [build_tree(r) for r in roots]
    all_nodes = []
    def get_all_nodes(node):
        all_nodes.append(node)
        for child in node.children:
            get_all_nodes(child)
    for r in root_nodes:
        get_all_nodes(r)

    def count_descendants(node):
        return sum(1 + count_descendants(c) for c in node.children)

    pelvis = None
    for node in all_nodes:
        if len(node.children) >= 3:
            pelvis = node
            break
    if not pelvis:
        pelvis = all_nodes[0]

    rename_map = {}
    if convention == "UE5":
        rename_map[pelvis.name] = "pelvis"
    elif convention == "Mixamo":
        rename_map[pelvis.name] = "Hips"

    pelvis_children = pelvis.children
    pelvis_children_sorted_by_size = sorted(pelvis_children, key=count_descendants, reverse=True)
    spine_root = pelvis_children_sorted_by_size[0]
    remaining_legs = pelvis_children_sorted_by_size[1:]

    legs_sorted_x = sorted(remaining_legs, key=lambda c: c.head.x, reverse=True)
    thigh_l = legs_sorted_x[0] if len(legs_sorted_x) > 0 else None
    thigh_r = legs_sorted_x[1] if len(legs_sorted_x) > 1 else None

    current_spine = spine_root
    spine_idx = 1
    chest_split = None
    while current_spine:
        if convention == "UE5":
            rename_map[current_spine.name] = f"spine_{spine_idx:02d}"
        elif convention == "Mixamo":
            rename_map[current_spine.name] = "Spine" if spine_idx == 1 else f"Spine{spine_idx - 1}"
            
        if len(current_spine.children) >= 3:
            chest_split = current_spine
            break
        elif len(current_spine.children) == 1:
            current_spine = current_spine.children[0]
            spine_idx += 1
        else:
            break

    if chest_split:
        chest_children = chest_split.children
        chest_children_sorted_by_size = sorted(chest_children, key=count_descendants)
        neck_root = chest_children_sorted_by_size[0]
        remaining_arms = chest_children_sorted_by_size[1:]
        arms_sorted_x = sorted(remaining_arms, key=lambda c: c.head.x, reverse=True)
        clavicle_l = arms_sorted_x[0] if len(arms_sorted_x) > 0 else None
        clavicle_r = arms_sorted_x[1] if len(arms_sorted_x) > 1 else None
        
        current_neck = neck_root
        neck_idx = 1
        while current_neck:
            if len(current_neck.children) == 0:
                rename_map[current_neck.name] = "head" if convention == "UE5" else "Head"
                break
            else:
                if convention == "UE5":
                    rename_map[current_neck.name] = f"neck_{neck_idx:02d}"
                elif convention == "Mixamo":
                    rename_map[current_neck.name] = "Neck" if neck_idx == 1 else f"Neck{neck_idx - 1}"
                current_neck = current_neck.children[0]
                neck_idx += 1
                
        def rename_arm_chain(start_node, suffix):
            side_prefix = "Left" if suffix == "l" else "Right"
            if convention == "UE5":
                rename_map[start_node.name] = f"clavicle_{suffix}"
            elif convention == "Mixamo":
                rename_map[start_node.name] = f"{side_prefix}Shoulder"
                
            curr = start_node.children[0] if start_node.children else None
            if curr:
                if convention == "UE5":
                    rename_map[curr.name] = f"upperarm_{suffix}"
                elif convention == "Mixamo":
                    rename_map[curr.name] = f"{side_prefix}Arm"
                curr = curr.children[0] if curr.children else None
            if curr:
                if convention == "UE5":
                    rename_map[curr.name] = f"lowerarm_{suffix}"
                elif convention == "Mixamo":
                    rename_map[curr.name] = f"{side_prefix}ForeArm"
                curr = curr.children[0] if curr.children else None
            if curr:
                if convention == "UE5":
                    rename_map[curr.name] = f"hand_{suffix}"
                elif convention == "Mixamo":
                    rename_map[curr.name] = f"{side_prefix}Hand"
                fingers = curr.children
                if len(fingers) > 0:
                    thumb_node = min(fingers, key=lambda f: f.head.y)
                    other_fingers = [f for f in fingers if f != thumb_node]
                    if suffix == "l":
                        other_fingers_sorted = sorted(other_fingers, key=lambda f: f.head.x, reverse=True)
                    else:
                        other_fingers_sorted = sorted(other_fingers, key=lambda f: f.head.x, reverse=False)
                    fingers_sorted = [thumb_node] + other_fingers_sorted
                else:
                    fingers_sorted = []
                    
                finger_names_ue5 = ["thumb", "index", "middle", "ring", "pinky"]
                finger_names_mixamo = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                if len(fingers_sorted) == 3:
                    finger_names_ue5 = ["thumb", "index", "pinky"]
                    finger_names_mixamo = ["Thumb", "Index", "Pinky"]
                    
                for f_idx, finger in enumerate(fingers_sorted):
                    curr_finger = finger
                    joint_idx = 1
                    while curr_finger:
                        if convention == "UE5":
                            name_prefix = finger_names_ue5[f_idx] if f_idx < len(finger_names_ue5) else f"finger_{f_idx}"
                            rename_map[curr_finger.name] = f"{name_prefix}_{joint_idx:02d}_{suffix}"
                        elif convention == "Mixamo":
                            name_prefix = finger_names_mixamo[f_idx] if f_idx < len(finger_names_mixamo) else f"Finger{f_idx}"
                            rename_map[curr_finger.name] = f"{side_prefix}Hand{name_prefix}{joint_idx}"
                        curr_finger = curr_finger.children[0] if curr_finger.children else None
                        joint_idx += 1
                        
        if clavicle_l:
            rename_arm_chain(clavicle_l, "l")
        if clavicle_r:
            rename_arm_chain(clavicle_r, "r")

    def rename_leg_chain(start_node, suffix):
        side_prefix = "Left" if suffix == "l" else "Right"
        if convention == "UE5":
            rename_map[start_node.name] = f"thigh_{suffix}"
        elif convention == "Mixamo":
            rename_map[start_node.name] = f"{side_prefix}UpLeg"
            
        curr = start_node.children[0] if start_node.children else None
        if curr:
            if convention == "UE5":
                rename_map[curr.name] = f"calf_{suffix}"
            elif convention == "Mixamo":
                rename_map[curr.name] = f"{side_prefix}Leg"
            curr = curr.children[0] if curr.children else None
        if curr:
            if convention == "UE5":
                rename_map[curr.name] = f"foot_{suffix}"
            elif convention == "Mixamo":
                rename_map[curr.name] = f"{side_prefix}Foot"
            curr = curr.children[0] if curr.children else None
        if curr:
            if convention == "UE5":
                rename_map[curr.name] = f"ball_{suffix}"
            elif convention == "Mixamo":
                rename_map[curr.name] = f"{side_prefix}ToeBase"
                
    if thigh_l:
        rename_leg_chain(thigh_l, "l")
    if thigh_r:
        rename_leg_chain(thigh_r, "r")

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    for orig, new in rename_map.items():
        if orig in armature.data.edit_bones:
            armature.data.edit_bones[orig].name = new
            
    # Inject root bone ONLY for UE5
    if convention == "UE5" and "pelvis" in armature.data.edit_bones:
        root_bone = armature.data.edit_bones.new("root")
        root_bone.head = (0.0, 0.0, 0.0)
        root_bone.tail = (0.0, 0.0, 0.1)
        armature.data.edit_bones["pelvis"].parent = root_bone
        
    bpy.ops.object.mode_set(mode='OBJECT')

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for orig, new in rename_map.items():
                if orig in obj.vertex_groups:
                    obj.vertex_groups[orig].name = new
                    
    if format_type == "glb":
        bpy.ops.export_scene.gltf(filepath=output_file)
    elif format_type == "fbx":
        bpy.ops.export_scene.fbx(filepath=output_file, use_selection=False, add_leaf_bones=False)
    print("Done!")

if __name__ == '__main__':
    main()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        temp_script = f.name
        
    try:
        subprocess.run(["blender", "--background", "--python", temp_script, "--", input_file, output_file, format_type, convention], check=True)
    finally:
        os.remove(temp_script)
        



def start_bpy_server(use_blender: bool = False):
    if use_blender:
        args = ["blender", "--background", "--python", "bpy_server.py"]
    else:
        args = [sys.executable, "bpy_server.py"]
        
    popen_kwargs = dict(
        args=args,
        stdout=None,
        stderr=None,
    )
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(**popen_kwargs)
    print(f"[Main] bpy_server.py started via {'Blender' if use_blender else 'Python'} (pid={proc.pid})")

    def cleanup():
        print(f"[Main] Terminating bpy_server.py (pid={proc.pid})")
        try:
            if proc.poll() is not None:
                return
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    atexit.register(cleanup)
    return proc


model = None
tokenizer = None
transform = None
CURRENT_MODEL_CKPT: Optional[str] = None
CURRENT_HF_PATH: Optional[str] = None


def load_model(model_ckpt: str, hf_path: Optional[str]) -> Tuple[str, str]:
    global model, tokenizer, transform, CURRENT_MODEL_CKPT, CURRENT_HF_PATH
    if hf_path == "None":
        hf_path = None
    if model is not None and model_ckpt == CURRENT_MODEL_CKPT and hf_path == CURRENT_HF_PATH:
        return ("Model already loaded.", model_ckpt)

    if not model_ckpt:
        raise RuntimeError("model_ckpt is empty. Please select a checkpoint.")

    print(f"Loading model: {model_ckpt}, hf_path={hf_path}")
    model = get_model(model_ckpt, hf_path=hf_path)
    assert model.tokenizer_config is not None
    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])
    CURRENT_MODEL_CKPT = model_ckpt
    CURRENT_HF_PATH = hf_path
    return ("Model loaded.", model_ckpt)


SUPPORTED_EXT = {".obj", ".fbx", ".glb"}


def collect_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    files = []
    for p in input_path.rglob("*"):
        if p.suffix.lower() in SUPPORTED_EXT:
            files.append(p)
    return files


def map_output_path(
    in_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    rel = in_path.relative_to(input_root)
    return (output_root / rel).with_suffix(".glb")


def post_bpy_payload(endpoint: str, payload):
    payload_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"skintokens_{endpoint}_", suffix=".pt", delete=False) as f:
            f.write(object_to_bytes(payload))
            payload_path = f.name
        request_payload = {"payload_path": payload_path}
        response = requests.post(
            f"{BPY_SERVER}/{endpoint}",
            data=object_to_bytes(request_payload),
        )
        response.raise_for_status()
        result = bytes_to_object(response.content)
        if isinstance(result, dict) and result.get("error") is not None:
            raise RuntimeError(result.get("traceback") or result["error"])
        return result
    finally:
        if payload_path is not None:
            try:
                os.remove(payload_path)
            except OSError:
                pass


def run_rig(
    filepaths: List[Path],
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    num_beams: int,
    use_skeleton: bool,
    use_transfer: bool,
    use_postprocess: bool,
    output_paths: List[Path],
    model_ckpt: str,
    hf_path: Optional[str],
    rename_ue5: bool = False,
):
    assert len(filepaths) == len(output_paths)

    load_model(model_ckpt, hf_path)

    datapath = {
        "data_name": None,
        "loader": "bpy_server",
        "filepaths": {"articulation": [str(p) for p in filepaths]},
    }

    dataset_config = DatasetConfig.parse(
        shuffle=False,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        datapath=datapath,
    ).split_by_cls()

    module = RigDatasetModule(
        predict_dataset_config=dataset_config,
        predict_transform=transform,
        tokenizer=tokenizer,
        process_fn=model._process_fn,
    )

    dataloader = module.predict_dataloader()["articulation"]

    results_out = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = {
            k: v.to("cuda") if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

        if not use_skeleton:
            batch.pop("skeleton_tokens", None)
            batch.pop("skeleton_mask", None)

        batch["generate_kwargs"] = dict(
            max_length=2048,
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            num_return_sequences=1,
            num_beams=int(num_beams),
            do_sample=True,
        )

        if "skeleton_tokens" in batch and "skeleton_mask" in batch:
            mask = batch["skeleton_mask"][0] == 1
            skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()
        else:
            skeleton_tokens = None

        preds: List[TokenRigResult] = model.predict_step(
            batch,
            skeleton_tokens=[skeleton_tokens] if skeleton_tokens is not None else None,
            make_asset=True,
        )["results"]

        asset = preds[0].asset
        assert asset is not None

        if use_postprocess:
            voxel = asset.voxel(resolution=196)
            asset.skin *= voxel_skin(
                grid=0,
                grid_coords=voxel.coords,
                joints=asset.joints,
                vertices=asset.vertices,
                faces=asset.faces,
                mode="square",
                voxel_size=voxel.voxel_size,
            )
            asset.normalize_skin()

        out_path = output_paths[i]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if use_transfer:
            payload = dict(
                source_asset=asset,
                target_path=asset.path,
                export_path=str(out_path),
                group_per_vertex=4,
            )
            res = post_bpy_payload("transfer", payload)
        else:
            payload = dict(
                asset=asset,
                filepath=str(out_path),
                group_per_vertex=4,
            )
            res = post_bpy_payload("export", payload)

        if res != "ok":
            print(f"[Error] {res}")
        else:
            print(f"[OK] Exported: {out_path}")
            # Determine renaming convention
            convention = "None"
            if isinstance(rename_ue5, bool):
                if rename_ue5:
                    convention = "UE5"
            elif isinstance(rename_ue5, str):
                convention = rename_ue5
                
            if convention in ["UE5", "Mixamo"]:
                suffix_type = out_path.suffix.lower().lstrip(".")
                if suffix_type in ["fbx", "glb"]:
                    try:
                        rename_joints_in_blender(str(out_path), str(out_path), suffix_type, convention)
                    except Exception as ex:
                        print(f"[Error] Failed to rename bones to {convention}: {ex}")

        results_out.append(out_path)

    return results_out


def run_cli(args):
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    files = collect_files(input_path)
    if not files:
        raise RuntimeError("No valid 3D files found.")

    if len(files) == 1 and output_path.suffix:
        outputs = [output_path]
    else:
        outputs = [
            map_output_path(f, input_path, output_path)
            for f in files
        ]

    run_rig(
        files,
        args.top_k,
        args.top_p,
        args.temperature,
        args.repetition_penalty,
        args.num_beams,
        args.use_skeleton,
        args.use_transfer,
        args.use_postprocess,
        outputs,
        args.model_ckpt,
        args.hf_path,
        rename_ue5=args.rename_ue5,
    )


TOT = 0
def run_gradio(
    files,
    top_k,
    top_p,
    temperature,
    repetition_penalty,
    num_beams,
    use_skeleton,
    use_transfer,
    use_postprocess,
    model_ckpt,
    hf_path,
    export_options,
    rename_ue5,
):
    if not files:
        return "Please upload at least one 3D model.", None

    tmp_out = Path(tempfile.mkdtemp(prefix="tokenrig_"))
    filepaths = [Path(f.name) for f in files]
    global TOT
    outputs = []
    for filepath in filepaths:
        TOT += 1
        outputs.append(tmp_out / f"res_{TOT}{export_options}")

    run_rig(
        filepaths,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        num_beams,
        use_skeleton,
        use_transfer,
        use_postprocess,
        outputs,
        model_ckpt,
        hf_path,
        rename_ue5=rename_ue5,
    )

    return f"Processed {len(outputs)} models.", [str(p) for p in outputs]


def launch_gradio():
    model_ckpts = MODEL_CKPTS
    hf_paths = HF_PATHS
    default_ckpt = model_ckpts[0] if model_ckpts else ""
    default_hf = hf_paths[0] if hf_paths else "None"

    with gr.Blocks(title="TokenRig Demo") as demo:
        gr.Markdown("## TokenRig Demo")
        gr.Markdown("Upload 3D assets, configure parameters, generate rigged GLB")

        files = gr.File(
            label="3D Models",
            file_count="multiple",
            file_types=[".obj", ".fbx", ".glb"],
        )

        with gr.Accordion("Generation Parameters", open=True):
            model_ckpt = gr.Dropdown(
                choices=model_ckpts,
                value=default_ckpt,
                label="Model checkpoint",
                interactive=True,
            )
            hf_path = gr.Dropdown(
                choices=hf_paths,
                value=default_hf,
                label="HF path",
                interactive=True,
            )
            top_k = gr.Slider(1, 200, value=5, step=1, label="top_k")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="temperature")
            repetition_penalty = gr.Slider(0.5, 3.0, value=2.0, step=0.1, label="repetition_penalty")
            num_beams = gr.Slider(1, 20, value=10, step=1, label="num_beams")
            use_skeleton = gr.Checkbox(False, label="Use skeleton (only generate skin if skeleton exists)")
            use_transfer = gr.Checkbox(False, label="Use transfer (maintain texture)")
            use_postprocess = gr.Checkbox(False, label="Use postprocess (voxel skin)")
            export_options = gr.Radio(choices=[".glb", ".fbx", ".obj"], value = ".glb", label="Export Format")
            rename_convention = gr.Radio(choices=["None", "UE5", "Mixamo"], value="None", label="Bone Naming Convention (GLB & FBX)")

        run_btn = gr.Button("Run", variant="primary")
        load_btn = gr.Button("Load Model")
        log = gr.Textbox(label="Status")
        output = gr.File(label="Generated Model", interactive=False)

        load_btn.click(
            lambda ckpt, hf: load_model(ckpt, hf)[0],
            inputs=[model_ckpt, hf_path],
            outputs=[log],
        )

        run_btn.click(
            run_gradio,
            inputs=[
                files,
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                num_beams,
                use_skeleton,
                use_transfer,
                use_postprocess,
                model_ckpt,
                hf_path,
                export_options,
                rename_convention,
            ],
            outputs=[log, output],
        )

    demo.launch(server_port=1024)

def wait_for_bpy_server(timeout=30):
    t0 = time.time()
    while True:
        try:
            requests.get(f"{BPY_SERVER}/ping", timeout=1)
            print("[Main] bpy_server is ready")
            return
        except Exception:
            if time.time() - t0 > timeout:
                raise RuntimeError("bpy_server failed to start")
            time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TokenRig Demo")
    parser.add_argument("--input", help="Input file or directory")
    parser.add_argument("--output", help="Output file or directory")

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    parser.add_argument("--num_beams", type=int, default=10)

    parser.add_argument("--use_skeleton", action="store_true")
    parser.add_argument("--use_transfer", action="store_true")
    parser.add_argument("--use_postprocess", action="store_true")
    parser.add_argument("--rename_ue5", action="store_true", help="Rename bones to UE5 convention")

    parser.add_argument("--model_ckpt", default=MODEL_CKPTS[0] if MODEL_CKPTS else "")
    parser.add_argument("--hf_path", default=None)

    parser.add_argument("--gradio", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Start without spawning a local bpy_server")

    args = parser.parse_args()

    if args.headless:
        server_proc = start_bpy_server(use_blender=True)
        wait_for_bpy_server()
    else:
        server_proc = start_bpy_server(use_blender=False)
        wait_for_bpy_server()

    if args.gradio or not args.input:
        launch_gradio()
    else:
        run_cli(args)
