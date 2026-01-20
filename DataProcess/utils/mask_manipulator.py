import os
import re
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_record_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("record_")


# ---------- numeric parsing & sorting ----------
_num_pat = re.compile(r"(\d+)")


def extract_last_digits(stem: str) -> Optional[int]:
    """Extract the last segment of consecutive digits from the file stem as index; return None if not found"""
    m = list(_num_pat.finditer(stem))
    return int(m[-1].group(1)) if m else None


def numeric_key(path: Path):
    """Sort key: prioritize by last digit segment (numeric), otherwise sort by dictionary order"""
    idx = extract_last_digits(path.stem)
    return (0, idx) if idx is not None else (1, path.stem)


# ---------- basic utilities ----------


def list_imgs(p: Path) -> List[Path]:
    if not p.exists():
        return []
    return [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")]


def find_min_rgb_in_cam(cam_dir: Path) -> Optional[Path]:
    """Return the jpg/jpeg path with the smallest numeric index in the specified cam_dir/color"""
    color_dir = cam_dir / "color"
    imgs = list_imgs(color_dir)
    if not imgs:
        return None
    imgs.sort(key=numeric_key)
    return imgs[0]


def find_min_rgb_in_record(record_dir: Path) -> Optional[Path]:
    """Return the jpg/jpeg path with the smallest numeric index in the first cam_dir/color under this record (backward compatible, returns first camera if multiple exist)"""
    cam_dirs = sorted([d for d in record_dir.iterdir() if d.is_dir() and d.name.startswith("cam")])
    if len(cam_dirs) == 0:
        return None
    cam_dir = cam_dirs[0]  # If multiple cameras exist, return the first one
    return find_min_rgb_in_cam(cam_dir)


# ---------- manifest read/write ----------


def dump_manifest(manifest_path: Path, entries: List[Dict[str, Any]]):
    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """Return a mapping of copied_stem -> entry"""
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {e["copied_stem"]: e for e in data if "copied_stem" in e}


# ---------- fallback parsing (when manifest is unavailable) ----------


def parse_mask_name_record(mask_name: str, record_names: List[str]) -> Optional[Tuple[str, str]]:
    """
    Infer record_name and "original basename" (without extension) from mask filename.
    规则：
      1) Remove extension; if starts with 'records_', remove that prefix;
      2) Use existing record_names for longest prefix match: <record_name>_xxxxx
    Return (record_name, rest_stem) or None
    """
    stem = Path(mask_name).stem
    if stem.startswith("records_"):
        stem = stem[len("records_"):]
    for rn in sorted(record_names, key=len, reverse=True):
        prefix = rn + "_"
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            return rn, Path(rest).stem
    return None


# ---------- subcommand: extract ----------


def first_frame_extract(records_dir, first_frames_dir: str = "data_for_mask"):
    """
    Extract the minimum index RGB from each cam_* folder in each record_*, copy to records/<first_frames_dir>_<cam_suffix>/,
    naming: records_<record_name>_<orig_stem>.jpg, and write manifest.json (including frame_index).
    Create separate folders for each camera, folder name: data_for_mask_<cam_suffix>.
    """
    records_dir = Path(records_dir).expanduser().resolve()
    if not records_dir.exists():
        raise FileNotFoundError(f"[Error] records root directory not found: {records_dir}")

    records = sorted([p for p in records_dir.iterdir() if is_record_dir(p)])
    if not records:
        print(f"[Warning] no record_* found in {records_dir}")
        raise FileNotFoundError(f"[Error] no record_* found in {records_dir}")

    # Store manifest entries for each camera {cam_suffix: [entries]}
    cam_manifests: Dict[str, List[Dict[str, Any]]] = {}
    cam_copied_counts: Dict[str, int] = {}

    for rec in records:
        # Find all folders starting with "cam"
        cam_dirs = sorted([d for d in rec.iterdir() if d.is_dir() and d.name.startswith("cam")])
        if len(cam_dirs) == 0:
            print(f"[Skip] {rec.name}: no folders starting with 'cam' found")
            continue

        # Process each camera
        for cam_dir in cam_dirs:
            rgb_path = find_min_rgb_in_cam(cam_dir)
            if rgb_path is None:
                print(f"[Skip] {rec.name}/{cam_dir.name}: missing color folder or no jpg/jpeg")
                continue

            # Extract camera suffix (remove cam_ prefix)
            cam_suffix = cam_dir.name[4:] if cam_dir.name.startswith("cam_") else cam_dir.name
            if not cam_suffix:
                print(f"[Warning] {rec.name}/{cam_dir.name}: cannot extract camera suffix, skipping")
                continue

            # Create separate folder for each camera
            data_for_mask_cam = records_dir / f"{first_frames_dir}_{cam_suffix}"
            if cam_suffix not in cam_manifests:
                ensure_dir(data_for_mask_cam)
                cam_manifests[cam_suffix] = []
                cam_copied_counts[cam_suffix] = 0

            orig_stem = rgb_path.stem
            frame_index = extract_last_digits(orig_stem)
            if frame_index is None:
                print(f"[Warning] {rec.name}/{cam_dir.name}: cannot extract number index from {orig_stem}, skipping")
                continue

            copied_name = f"records_{rec.name}_{orig_stem}.jpg"
            dst = data_for_mask_cam / copied_name
            shutil.copy2(rgb_path, dst)

            rel_src = rgb_path.relative_to(records_dir)
            cam_manifests[cam_suffix].append({
                "record_name": rec.name,
                "cam_name": cam_dir.name,
                "orig_stem": orig_stem,
                "frame_index": int(frame_index),             # Key: save frame index
                "source_rgb_rel": str(rel_src),              # 'record_xx/cam_xxx/color/frame_00006.jpg'
                "copied_name": copied_name,                  # 'records_record_xx_frame_00006.jpg'
                "copied_stem": Path(copied_name).stem        # 'records_record_xx_frame_00006'
            })
            print(f"[OK] {rec.name}/{cam_dir.name}: {rgb_path.name} (idx={frame_index}) -> {first_frames_dir}_{cam_suffix}/{copied_name}")
            cam_copied_counts[cam_suffix] += 1

    # Write separate manifest.json for each camera
    total_copied = 0
    for cam_suffix, manifest_entries in cam_manifests.items():
        data_for_mask_cam = records_dir / f"{first_frames_dir}_{cam_suffix}"
        dump_manifest(data_for_mask_cam / "manifest.json", manifest_entries)
        print(f"[OK] camera {cam_suffix}: extracted {cam_copied_counts[cam_suffix]} minimum index RGB to {data_for_mask_cam}, wrote manifest.json")
        total_copied += cam_copied_counts[cam_suffix]
    
    print(f"[OK] total extracted {total_copied} minimum index RGB, involving {len(cam_manifests)} cameras")


# ---------- subcommand: restore ----------


def masks_restore(records_dir, organized_dir="organized", first_frames_dir="data_for_mask"):
    """
    Restore PNG masks under mask_i to respective record_*/cam_xxx/organized_dir/masks_i/.
    Now masks_x folder is inside data_for_mask_xxx folder, need to find corresponding cam_xxx folder based on camera suffix.
    Use frame_index from manifest → `%016d.png`.
    If no manifest entry, fallback parse record_name and number index from filename.
    """
    records_dir = Path(records_dir).expanduser().resolve()
    if not records_dir.exists():
        raise FileNotFoundError(f"[Error] records root directory not found: {records_dir}")

    # Index record directories
    record_dirs = {p.name: p for p in records_dir.iterdir() if is_record_dir(p)}
    record_names = list(record_dirs.keys())
    data_for_mask_dirs = sorted([p for p in records_dir.iterdir() 
                                 if p.is_dir() and p.name.startswith(first_frames_dir + "_")])
    
    if not data_for_mask_dirs:
        # Try to find data_for_mask folder (directly in records_dir)
        old_data_for_mask = records_dir / first_frames_dir
        if old_data_for_mask.exists():
            data_for_mask_dirs = [old_data_for_mask]
        else:
            raise FileNotFoundError(f"[Error] no {first_frames_dir}_xxx or {first_frames_dir} found in {records_dir}")

    total_placed = 0
    
    # Iterate over each data_for_mask_xxx folder
    for data_for_mask_dir in data_for_mask_dirs:
        if data_for_mask_dir.name.startswith(first_frames_dir + "_"):
            cam_suffix = data_for_mask_dir.name[len(first_frames_dir) + 1:]  # Remove "data_for_mask_" prefix
        else:
            cam_suffix = None
            print(f"[INFO] detected old structure folder: {data_for_mask_dir.name}")

        masks_folders = sorted([p for p in data_for_mask_dir.iterdir() 
                               if p.is_dir() and p.name.startswith("mask")])
        
        if not masks_folders:
            print(f"[Skip] no folders starting with 'mask' found in {data_for_mask_dir.name}")
            continue

        # Load manifest for corresponding camera
        manifest_path = data_for_mask_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"[Warning] {manifest_path} does not exist, will use filename fallback parsing")
            manifest = {}
        else:
            manifest = load_manifest(manifest_path)

        # Process each mask folder
        for mask_dir in masks_folders:
            index = mask_dir.name.split("_")[-1]
            if not index or not index.isdigit():
                print(f"[Warning] cannot extract number index from {mask_dir.name}, skipping")
                continue
            
            print(f"[INFO] restoring {data_for_mask_dir.name}/{mask_dir.name} (index={index}, cam_suffix={cam_suffix}) ...")
            masks = sorted([p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
            if not masks:
                print(f"[Warning] no .png found in {mask_dir}, skipping")
                continue

            placed = 0
            for mask in masks:
                mask_stem = mask.stem
                record_name: Optional[str] = None
                frame_index: Optional[int] = None
                cam_name: Optional[str] = None

                if mask_stem in manifest:
                    entry = manifest[mask_stem]
                    record_name = entry["record_name"]
                    frame_index = int(entry["frame_index"])
                    # If manifest has cam_name, use it; otherwise build based on cam_suffix
                    cam_name = entry.get("cam_name")
                    if not cam_name and cam_suffix:
                        cam_name = f"cam_{cam_suffix}"
                    print(f"[MAP] {mask.name} -> record={record_name}, cam={cam_name}, idx={frame_index}")
                else:
                    # No manifest: fallback parse from filename
                    parsed = parse_mask_name_record(mask.name, record_names)
                    if parsed is None:
                        print(f"[Skip] cannot parse record name (and manifest does not exist): {mask.name}")
                        continue
                    record_name, rest_stem = parsed
                    idx = extract_last_digits(rest_stem)
                    if idx is None:
                        print(f"[Skip] cannot extract frame index from {mask.name} (and manifest does not exist)")
                        continue
                    frame_index = int(idx)
                    # If no cam_name, build based on cam_suffix
                    if cam_suffix:
                        cam_name = f"cam_{cam_suffix}"
                    print(f"[FALLBACK] {mask.name} -> record={record_name}, cam={cam_name}, idx={frame_index}")

                if record_name not in record_dirs:
                    print(f"[Skip] record directory not found: {record_name} (mask={mask.name})")
                    continue

                # Determine target cam folder
                if cam_name:
                    target_cam_dir = record_dirs[record_name] / cam_name
                    if not target_cam_dir.exists():
                        print(f"[Warning] camera directory not found: {target_cam_dir}, skipping {mask.name}")
                        continue
                    target_masks_dir = target_cam_dir / organized_dir / f"masks_{index}"
                    target_rgb_dir = target_cam_dir / organized_dir / "rgb"
                else:
                    target_masks_dir = record_dirs[record_name] / organized_dir / f"masks_{index}"
                    target_rgb_dir = record_dirs[record_name] / organized_dir / "rgb"

                os.makedirs(target_masks_dir, exist_ok=True)
                dst_name = f"{frame_index:016d}.png"
                dst = target_masks_dir / dst_name

                # Optional consistency check: corresponding organized/rgb/%016d.png exists
                org_rgb = target_rgb_dir / f"{frame_index:016d}.png"
                if not org_rgb.exists():
                    print(f"[WARN] {org_rgb} does not exist, but still write to {dst_name}")

                shutil.copy2(mask, dst)
                target_path_str = str(target_masks_dir.relative_to(records_dir))
                print(f"[OK] restored {target_path_str}/{dst_name}")
                placed += 1
                total_placed += 1

            print(f"[OK] {data_for_mask_dir.name}/{mask_dir.name}: restored {placed} mask files")
    
    print(f"[OK] total restored {total_placed} mask files")
