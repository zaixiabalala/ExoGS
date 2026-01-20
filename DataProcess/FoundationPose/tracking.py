import os
from pathlib import Path
from run_batch_pose_tracking import batch_pose_tracking


def get_object_num(records_dir: Path):
    # Find all record_* directories
    record_dirs = [Path(records_dir) / d for d in os.listdir(records_dir) 
                   if os.path.isdir(os.path.join(records_dir, d)) and d.startswith("record")]

    obj_num = 0
    if len(record_dirs) > 0:
        first_record = record_dirs[0]
        # Find cam_* folders
        cam_dirs = sorted([d for d in first_record.iterdir() if d.is_dir() and d.name.startswith("cam")])
        if len(cam_dirs) > 0:
            # Find the organized folder in the first cam folder
            organized_dir = cam_dirs[0] / "organized"
            if organized_dir.exists():
                obj_num = len([d for d in os.listdir(organized_dir) 
                             if os.path.isdir(os.path.join(organized_dir, d)) and d.startswith("mask")])
        # If the new structure is not found, try the old structure (backward compatible)
        if obj_num == 0:
            old_organized = first_record / "organized"
            if old_organized.exists():
                obj_num = len([d for d in os.listdir(old_organized) 
                             if os.path.isdir(os.path.join(old_organized, d)) and d.startswith("mask")])
    
    if obj_num == 0:
        raise ValueError(f"[ERROR] cannot find organized folder or mask directory in {records_dir}")
    
    print(f"[INFO] found {obj_num} objects (mask directories)")
    return obj_num

def main():
    ############################################################

    debug = 3
    Origin_Data_dir = "/home/ubuntu/data/Origin_Data"
    records_name = "records_unscrew_cap_1221"

    ###########################################################
    
    records_dir = Path(Origin_Data_dir) / records_name

    for i in range(get_object_num(records_dir)):
        print(f"[INFO] starting to process object idx={i+1}")
        batch_pose_tracking(idx=(i+1), records_root=str(records_dir), debug=debug)

if __name__ == "__main__":
    main()