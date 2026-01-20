import os
import re


def rename_remaining_files(angles_dir, file_extension):
    all_files = [f for f in os.listdir(angles_dir) if f.endswith(file_extension)]
    if not all_files:
        print(f"[Warning] no {file_extension} files found in {angles_dir}")
        return []
    sorted_files = sorted(all_files, key=extract_number_from_filename)

    # rename files
    for i, old_filename in enumerate(sorted_files):
        name_without_ext, file_ext = os.path.splitext(old_filename)

        match = re.search(r'(\d+)(?!.*\d)', name_without_ext)
        if match:
            number_part = match.group(1)
            non_number_part = name_without_ext[:match.start()]
            
            original_number = int(number_part)
            new_number = i  
            new_number_str = f"{new_number:0{len(number_part)}d}"
            new_filename = f"{non_number_part}{new_number_str}{file_ext}"
            
            old_path = os.path.join(angles_dir, old_filename)
            new_path = os.path.join(angles_dir, new_filename)
            
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_filename} -> {new_filename}")


def extract_number_from_filename(filename):
    match = re.search(r'(\d+)(?:\.[a-zA-Z0-9]+)?$', filename)
    if match:
        return int(match.group(1))
    return -1


def get_files_to_delete(angles_dir, file_extension, pre_frames):
    all_files = [f for f in os.listdir(angles_dir) if f.endswith(file_extension)]
    if not all_files:
        print(f"[Warning] no {file_extension} files found in {angles_dir}")
        return []
    sorted_files = sorted(all_files, key=extract_number_from_filename)
    return sorted_files[:pre_frames]


def cut_pre_frames(records_dir, pre_frames):
    assert os.path.exists(records_dir), f"records_dir not found: {records_dir}"

    record_dirs = [f for f in os.listdir(records_dir)
                    if os.path.isdir(os.path.join(records_dir, f)) and f.startswith('record_')]
    assert len(record_dirs) > 0, f"no record_dirs found: {record_dirs}"

    for record_dir in record_dirs:
        print("[Info] processing record_dir: ", record_dir)
        # find all folders starting with "cam" in record_dir folder
        record_path = os.path.join(records_dir, record_dir)
        cam_dirs = [d for d in os.listdir(record_path)
                    if os.path.isdir(os.path.join(record_path, d)) and d.startswith("cam")]
        if not cam_dirs:
            raise FileNotFoundError(f"[Warning] No cam* folders found in {record_path}")

        for cam_dir_name in cam_dirs:
            color_dir = os.path.join(record_path, cam_dir_name, "color")
            depth_dir = os.path.join(record_path, cam_dir_name, "depth")

            if os.path.exists(color_dir):
                files_to_delete = get_files_to_delete(color_dir, ".png", pre_frames)
                # delete pre_frames .png files from color_dir
                for file in files_to_delete:
                    file_path = os.path.join(color_dir, file)
                    os.remove(file_path)
                print(f"Deleted {len(files_to_delete)} .png files from color_dir")
                rename_remaining_files(color_dir, ".png")

            if os.path.exists(depth_dir):
                files_to_delete = get_files_to_delete(depth_dir, ".png", pre_frames)
                # delete pre_frames .png files from depth_dir
                for file in files_to_delete:
                    file_path = os.path.join(depth_dir, file)
                    os.remove(file_path)
                print(f"Deleted {len(files_to_delete)} .png files from depth_dir")
                rename_remaining_files(depth_dir, ".png")

        angles_dir = os.path.join(record_path, "angles")
        if os.path.exists(angles_dir):
            files_to_delete = get_files_to_delete(angles_dir, ".npy", pre_frames)
            for file in files_to_delete:
                file_path = os.path.join(angles_dir, file)
                os.remove(file_path)
            print(f"Deleted {len(files_to_delete)} .npy files from angles_dir")
            rename_remaining_files(angles_dir, ".npy")

        tcps_dir = os.path.join(record_path, "tcps")
        if os.path.exists(tcps_dir):
            files_to_delete = get_files_to_delete(tcps_dir, ".npy", pre_frames)
            for file in files_to_delete:
                file_path = os.path.join(tcps_dir, file)
                os.remove(file_path)
            print(f"Deleted {len(files_to_delete)} .npy files from tcps_dir")
            rename_remaining_files(tcps_dir, ".npy")
