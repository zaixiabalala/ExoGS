import os
from utils.mask_manipulator import masks_restore

#########################################################

Origin_Data_dir = "/home/ubuntu/data/Origin_Data"
records_name = "records_unscrew_cap_1221"

#########################################################

records_dir = os.path.join(Origin_Data_dir, records_name)
assert os.path.exists(records_dir), f"records_dir not found: {records_dir}"

# backfill masks to organized
print(f"[Info] Backfilling masks to organized...")
masks_restore(records_dir)