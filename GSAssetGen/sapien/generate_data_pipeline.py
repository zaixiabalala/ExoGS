import os
import subprocess
from utils.get_images_from_urdf import render_multiview_images


def main():
    # Object list to process
    urdf_name_list = ["square_orange_cap"]
    
    # Output directory for generated data
    sapien_data_dir = "/home/ubuntu/data/sapien_data"

    # Path to COLMAP format conversion script
    txt_2_bin_script = "./utils/txt_2_bin.py"

    for urdf_name in urdf_name_list:
        raw_data_dir = os.path.join(sapien_data_dir, f"{urdf_name}")
        
        # Render multi-view images and generate COLMAP sparse reconstruction
        render_multiview_images(
            urdf_name, 
            raw_data_dir, 
            phi_range=[10, 190],        # Elevation angle range [min, max] in degrees
            theta_range=[0, 360],       # Azimuth angle range [min, max] in degrees
            phi_delta=10,               # Step size for elevation angle
            theta_delta=10,             # Step size for azimuth angle
            cam_distance=[0.3, 0.5],    # Camera distance random range [min, max] from object
            camera_width=800,           # Image width in pixels
            camera_height=800,          # Image height in pixels
            cover_mode=True,            # Enable coverage mode for better view distribution
            generate_colmap=True        # Automatically generate COLMAP sparse reconstruction
        )
        
        # Convert COLMAP format from TXT to BIN
        print(f"[INFO] Converting txt to bin for {urdf_name}")
        subprocess.run(
            ["python", txt_2_bin_script,
             "--input_model", raw_data_dir + "/sparse/0",
             "--input_format", ".txt",
             "--output_model", raw_data_dir + "/sparse/0",
             "--output_format", ".bin"]
        )

if __name__ == "__main__":
    main()