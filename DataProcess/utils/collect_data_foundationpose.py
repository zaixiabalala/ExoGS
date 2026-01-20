import os
import sys
import shutil
from pathlib import Path


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def find_records(records_root: Path):
    if not records_root.exists():
        raise FileNotFoundError(f"[Error] records root directory not found: {records_root}")
    return sorted([p for p in records_root.iterdir() if p.is_dir() and p.name.startswith("record_")])


def numeric_sort_key(p: Path):
    stem = p.stem
    digits = ''.join(ch for ch in stem if ch.isdigit())
    try:
        return int(digits) if digits else stem
    except:
        return stem


def copy_mesh_triple(mesh_dir: Path, out_mesh_dir: Path, mesh_name: str):
    ensure_dir(out_mesh_dir)

    # automatically scan files in the folder
    all_files = list(mesh_dir.iterdir())
    files_by_ext = {}

    # classify files by file extension
    for file_path in all_files:
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext not in files_by_ext:
                files_by_ext[ext] = []
            files_by_ext[ext].append(file_path)

    # verify necessary file types
    required = []
    obj_files = files_by_ext.get('.obj', [])
    mtl_files = files_by_ext.get('.mtl', [])
    png_files = files_by_ext.get('.png', [])

    # check obj file
    if len(obj_files) == 0:
        raise FileNotFoundError(f"[Error] .obj file not found: {mesh_dir}")
    elif len(obj_files) > 1:
        raise ValueError(f"[Error] multiple .obj files found: {[f.name for f in obj_files]}")
    # 检查mtl文件
    if len(mtl_files) == 0:
        raise FileNotFoundError(f"[Error] .mtl file not found: {mesh_dir}")
    elif len(mtl_files) > 1:
        raise ValueError(f"[Error] multiple .mtl files found: {[f.name for f in mtl_files]}")

    # add necessary files to required list
    required.extend([obj_files[0].name, mtl_files[0].name])

    # add all png files (optional)
    for png_file in png_files:
        required.append(png_file.name)

    print(f"[INFO] files detected: {required}")

    # copy files
    for f in required:
        src = mesh_dir / f
        if not src.exists():
            raise FileNotFoundError(f"[Error] file not found: {src}")
        shutil.copy2(src, out_mesh_dir / f)


def organize_one_record(
    rec_dir: Path,
    cam_dir: Path,
    organized_dir_name: str,
    cam_intrinsic_path: Path,
    object_mesh_dir_list: list[Path]
):
    # process single cam folder
    color_dir = cam_dir / "color"
    depth_dir = cam_dir / "depth"

    if not color_dir.exists() or not depth_dir.exists():
        print(f"[Skip] {cam_dir} missing color or depth folder")
        raise FileNotFoundError(f"[Error] {cam_dir} missing color or depth folder")

    color_list = sorted(color_dir.glob("*.png"), key=numeric_sort_key)
    depth_list = sorted(depth_dir.glob("*.png"), key=numeric_sort_key)
    assert len(color_list) == len(depth_list) and len(color_list) > 0, f"[Error] {cam_dir} color/depth count mismatch"

    # put organized folder inside cam folder
    out_dir = cam_dir / organized_dir_name
    out_rgb = out_dir / "rgb"
    out_depth = out_dir / "depth"
    ensure_dir(out_rgb)
    ensure_dir(out_depth)

    # copy uniform parameters (all organized_* are the same)
    shutil.copy2(cam_intrinsic_path, out_dir / "cam_intrinsic.txt")

    for i, object_mesh_dir in enumerate(object_mesh_dir_list):
        copy_mesh_triple(object_mesh_dir, out_dir / f"mesh_{i+1}", object_mesh_dir.name)

    # rename copied frames
    for i, (cpath, dpath) in enumerate(zip(color_list, depth_list)):
        name16 = f"{i:016d}"
        shutil.copy2(cpath, out_rgb / f"{name16}.png")
        shutil.copy2(dpath, out_depth / f"{name16}.png")

    with open(out_dir / "README_organized.txt", "w", encoding="utf-8") as f:
        f.write(
            "Auto-generated for FoundationPose inference.\n"
            "Structure:\n"
            "- depth/: uint16 PNG, 16-digit names from 0\n"
            "- rgb/:   PNG,      16-digit names from 0\n"
            "- masks/: (empty, user-provided 000...000.png later)\n"
            "- mesh_1/: (optional first object: box1.obj / .mtl / box1.png)\n"
            "- mesh_2/: (optional second object: box1.obj / .mtl / box1.png)\n"
            "- cam_intrinsic.txt (uniform across all organized_*)\n"
        )

    print(f"[OK] {rec_dir.name}/{cam_dir.name} -> {out_dir}")


def get_angles(records_root: Path):
    """
    check if each record directory contains an angles folder, if not create it
    and move all files starting with "angle" to the angles folder
    """
    recs = find_records(records_root)
    if not recs:
        print(f"[Error] no record_* found in {records_root}")
        return

    for rec in recs:
        angles_dir = rec / "angles"

        # check if angles folder exists
        if not angles_dir.exists():
            print(f"[Create] create angles folder in {rec.name}")
            ensure_dir(angles_dir)

        # find all files starting with "angle"
        angle_files = list(angles_dir.glob("angle*"))

        if angle_files:
            print(f"[Process] found {len(angle_files)} angle files in {rec.name}")

            # move all angle files to angles folder
            for angle_file in angle_files:
                if angle_file.is_file():  # ensure it is a file not a directory
                    dst_path = angles_dir / angle_file.name
                    if not dst_path.exists():  # avoid overwriting existing files
                        shutil.move(str(angle_file), str(dst_path))
                        print(f"  [Move] {angle_file.name} -> angles/")
                    else:
                        print(f"  [Skip] {angle_file.name} already exists in angles folder")
        else:
            print(f"[Info] no angle files found in {rec.name}")


def collect_organized_data(records_dir, 
                            organized_dir_name, 
                            object_name_list: list[str],
                            cam_intrinsic_path: Path = Path(os.path.join(os.path.dirname(__file__), "../assets", "cam_intrinsic.txt")).expanduser().resolve()):
    records_root = Path(records_dir).expanduser().resolve()
    assert object_name_list is not None, "[Error] object_name_list cannot be empty"

    object_mesh_dir_list = []
    for object_name in object_name_list:
        object_mesh_dir = Path(os.path.join(os.path.dirname(__file__), "../assets", object_name)).expanduser().resolve()
        if not object_mesh_dir.exists():
            print(f"[Error] {object_mesh_dir} does not exist")
            sys.exit(1)
        object_mesh_dir_list.append(object_mesh_dir)

    if not cam_intrinsic_path.exists():
        print(f"[Error] cam_intrinsic.txt does not exist: {cam_intrinsic_path}")
        sys.exit(1)

    recs = find_records(records_root)
    if not recs:
        print(f"[Error] no record_* found in {records_root}")
        sys.exit(1)

    for rec in recs:
        # find all folders starting with "cam" in rec_dir folder
        cam_dirs = sorted([d for d in rec.iterdir() if d.is_dir() and d.name.startswith("cam")])
        if len(cam_dirs) == 0:
            print(f"[Skip] no folders starting with 'cam' found in {rec}")
            continue
        
        # generate organized folder for each cam folder
        for cam_dir in cam_dirs:
            try:
                organize_one_record(
                    rec,
                    cam_dir=cam_dir,
                    organized_dir_name=organized_dir_name,
                    cam_intrinsic_path=cam_intrinsic_path,
                    object_mesh_dir_list=object_mesh_dir_list,
                )
            except Exception as e:
                print(f"[Error] {rec}/{cam_dir.name}: {e}")
