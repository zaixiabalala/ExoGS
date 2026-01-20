import os
from utils.cut_frames import cut_pre_frames
from utils.collect_data_foundationpose import collect_organized_data
from utils.mask_manipulator import first_frame_extract

#########################################################

# !!! Please backup the original data before running this script to avoid data loss !!!

# Before running this script, please make sure to 
# add the mesh files of the objects to the assets folder and 
# ensure the camera intrinsic file is in the assets folder.

Origin_Data_dir = "/home/ubuntu/data/Origin_Data"  # Records folder
records_name = "records_unscrew_cap_1221"
cut_pre_frames_num = 0    # Number of initial frames to remove
object_name_list = ["square_pink_can", "square_orange_cap"]

#########################################################

records_dir = os.path.join(Origin_Data_dir, records_name)
assert os.path.exists(records_dir), f"records_dir not found: {records_dir}"

# remove specified number of frames
print(f"[Info] Removing first {cut_pre_frames_num} frames...")
cut_pre_frames(records_dir, cut_pre_frames_num)

# generate organized folders
print(f"[Info] Generating organized folders...")
collect_organized_data(records_dir, "organized", object_name_list=object_name_list)

# generate data_for_mask folders
print(f"[Info] Generating data_for_mask folders...")
first_frame_extract(records_dir)
