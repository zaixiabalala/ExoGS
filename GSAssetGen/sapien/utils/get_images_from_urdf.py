import os
import time
import sapien
import shutil
import numpy as np
import psutil
import gc
from PIL import Image, ImageColor
from tqdm import tqdm
from utils.camera_utils import get_camera_pos_mat, get_cam_pos_fix
from utils.image_utils import change_image_background

# Go up two levels from utils/get_images_from_urdf.py to the sapien folder
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_scene():
    # Enable ray tracing renderer
    sapien.render.set_camera_shader_dir("rt")  # Camera uses ray tracing (rt)
    sapien.render.set_viewer_shader_dir("rt")  # Viewer uses ray tracing
    # Set ray tracing parameters
    sapien.render.set_ray_tracing_samples_per_pixel(256)  # Reduce samples to decrease GPU load
    sapien.render.set_ray_tracing_denoiser("optix")  # Use OptiX denoiser
    scene = sapien.Scene()  # Create scene
    scene.set_timestep(1 / 100.0)  # Set physics timestep

    # Set background color to dark (black) to improve contrast between object and background,
    # facilitating feature point extraction
    try:
        # Method 1: Through sapien.render module (if supported)
        if hasattr(sapien.render, 'set_clear_color'):
            sapien.render.set_clear_color([0.0, 0.0, 0.0])  # Black background
        # Method 2: Get renderer through scene
        elif hasattr(scene, 'get_renderer'):
            renderer = scene.get_renderer()
            if hasattr(renderer, 'set_clear_color'):
                renderer.set_clear_color([0.0, 0.0, 0.0])  # Black background
        # Method 3: Through scene.renderer attribute
        elif hasattr(scene, 'renderer'):
            if hasattr(scene.renderer, 'set_clear_color'):
                scene.renderer.set_clear_color([0.0, 0.0, 0.0])  # Black background
    except Exception as e:
        print(f"[Warning] Failed to set background color: {e}")

    return scene


def setup_lighting(scene):
    """Lighting setup for debugging"""
    # Ambient light
    scene.set_ambient_light([0.8, 0.8, 0.8])

    scene.add_directional_light(
        direction=[-0.5, -0.5, -1],
        color=[2.0, 2.0, 2.0],
        shadow=True,
        shadow_map_size=4096,
    )
    # Fill lights from four sides
    for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        scene.add_point_light(
            position=[x, y, 0],
            color=[0.7, 0.7, 0.7],
            shadow=False,
        )

def load_urdf(scene, urdf_name):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = os.path.join(current_dir, "assets/urdf", urdf_name + ".urdf")
    asset = loader.load(urdf_path)
    assert asset, "failed to load URDF."


def create_cameras_batch(scene, urdf_name, phi_range, theta_range, phi_delta, theta_delta,
                          cam_distance, width, height, batch_start, batch_end):
    """Create a batch of cameras"""
    cameras = []
    phi_values = list(range(phi_range[0], phi_range[1], phi_delta))
    theta_values = list(range(theta_range[0], theta_range[1], theta_delta))

    total_cameras = len(phi_values) * len(theta_values)
    cameras_in_batch = batch_end - batch_start

    for i in range(cameras_in_batch):
        camera_idx = batch_start + i
        if camera_idx >= total_cameras:
            break

        phi_idx = camera_idx // len(theta_values)
        theta_idx = camera_idx % len(theta_values)
        phi = phi_values[phi_idx]
        theta = theta_values[theta_idx]

        cam_pos = get_cam_pos_fix(
            phi,
            theta,
            np.random.uniform(cam_distance[0], cam_distance[1])
        )

        forward = -cam_pos / np.linalg.norm(cam_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos

        # Use elevation and azimuth angles as data name, required for subsequent model training
        data_name = f"{phi}_{theta}"

        camera = scene.add_camera(
            name=f"camera_{data_name}",
            width=width,
            height=height,
            fovy=np.deg2rad(35),  # 35 degree field of view
            near=0.1,
            far=100.0,
        )
        camera.entity.set_pose(sapien.Pose(mat44))
        cameras.append(camera)

    return cameras


def _save_depth_data(depth, output_path, data_name):
    """Save depth map data"""
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_normalized, mode='L')
    depth_image.save(os.path.join(output_path, f"{data_name}_depth.png"))
    np.save(os.path.join(output_path, f"{data_name}_depth.npy"), depth)
    del depth_image, depth_normalized


def _save_mask_image(mask, output_path, data_name):
    """Save mask image"""
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
    mask_image = Image.fromarray(color_palette[mask])
    mask_image.save(os.path.join(output_path, f"{data_name}_mask.png"))
    del mask_image, colormap, color_palette

def process_in_batches(urdf_name, phi_range, theta_range, phi_delta, theta_delta, cam_distance,
                        camera_width, camera_height, raw_data_dir, cover_mode,
                        save_rgb=True, save_depth=False, save_mask=False, save_npz=True):
    """Process in batches, recreating scene for each batch

    Args:
        save_rgb: Whether to save RGB images to images folder
        save_depth: Whether to save depth maps to depths folder
        save_mask: Whether to save mask images to masks folder
        save_npz: Whether to save npz data to npzs folder
    """
    batch_size = 30  # Reduce batch size
    phi_values = list(range(phi_range[0], phi_range[1], phi_delta))
    theta_values = list(range(theta_range[0], theta_range[1], theta_delta))
    total_cameras = len(phi_values) * len(theta_values)

    # Create output folders (in cover mode, these folders should be brand new)
    output_dirs = {}
    if save_rgb:
        output_dirs['rgb'] = os.path.join(raw_data_dir, 'images')
        # In cover mode, ensure folder is brand new
        if cover_mode and os.path.exists(output_dirs['rgb']):
            shutil.rmtree(output_dirs['rgb'])
        os.makedirs(output_dirs['rgb'], exist_ok=True)
    if save_depth:
        output_dirs['depth'] = os.path.join(raw_data_dir, 'depths')
        if cover_mode and os.path.exists(output_dirs['depth']):
            shutil.rmtree(output_dirs['depth'])
        os.makedirs(output_dirs['depth'], exist_ok=True)
    if save_mask:
        output_dirs['mask'] = os.path.join(raw_data_dir, 'masks')
        if cover_mode and os.path.exists(output_dirs['mask']):
            shutil.rmtree(output_dirs['mask'])
        os.makedirs(output_dirs['mask'], exist_ok=True)
    if save_npz:
        output_dirs['npz'] = os.path.join(raw_data_dir, 'npzs')
        if cover_mode and os.path.exists(output_dirs['npz']):
            shutil.rmtree(output_dirs['npz'])
        os.makedirs(output_dirs['npz'], exist_ok=True)

    total_batches = (total_cameras - 1) // batch_size + 1
    output_types = [t for t, enabled in [('RGB', save_rgb), ('Depth', save_depth),
                                         ('Mask', save_mask), ('NPZ', save_npz)] if enabled]
    print(f"[Info] Total cameras to process: {total_cameras}, processing in {total_batches} batches")
    print(f"[Info] Output configuration: {', '.join(output_types)}")

    # Overall progress bar
    batch_indices = list(range(0, total_cameras, batch_size))
    pbar_batches = tqdm(batch_indices,
                        desc="Batch progress",
                        unit="batch",
                        position=0,
                        leave=True,
                        total=total_batches)

    for batch_idx in pbar_batches:
        batch_end = min(batch_idx + batch_size, total_cameras)
        batch_num = batch_idx // batch_size + 1
        pbar_batches.set_description(f"Batch {batch_num}/{total_batches} "
                                     f"(cameras {batch_idx}-{batch_end-1})")

        # Recreate scene and cameras
        scene = set_scene()
        setup_lighting(scene)
        load_urdf(scene, urdf_name)
        cameras = create_cameras_batch(scene, urdf_name, phi_range, theta_range, phi_delta,
                                        theta_delta, cam_distance, camera_width, camera_height,
                                        batch_idx, batch_end)

        scene.step()
        scene.update_render()

        # Process this batch of cameras, using inner progress bar
        pbar_cameras = tqdm(cameras,
                            total=len(cameras),
                            desc="  Processing cameras",
                            unit="image",
                            position=1,
                            leave=False)

        for i, camera in enumerate(pbar_cameras):
            try:
                camera.take_picture()
                data_name = f"{camera.name.split('_')[1]}_{camera.name.split('_')[2]}"
                pbar_cameras.set_description(f"  Processing camera {data_name} "
                                             f"({i+1}/{len(cameras)})")

                # Get RGB data (if save_rgb or save_npz is needed)
                rgb = None
                mask = None
                if save_rgb or save_npz:
                    rgba = camera.get_picture("Color")  # [H, W, 4]
                    rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
                    del rgba

                    # If saving RGB, need to get mask for background replacement
                    if save_rgb:
                        seg_labels = camera.get_picture("Segmentation")
                        mask = seg_labels[..., 1].astype(np.uint8)
                        del seg_labels
                        # Replace background with dark background (black) to improve contrast
                        rgb_with_dark_bg, _ = change_image_background(
                            part_mask=mask,
                            rgb=rgb,
                            depth=np.zeros((rgb.shape[0], rgb.shape[1])),  # No depth needed, pass empty array
                            part_mask_id_min=2,
                            image_show=False,
                            background_color=[0.0, 0.0, 0.0]  # Black background
                        )
                        Image.fromarray(rgb_with_dark_bg).save(
                            os.path.join(output_dirs['rgb'], f"{data_name}_rgb.png"))
                        del rgb_with_dark_bg

                # Get mask data (if save_mask or save_npz is needed, and not obtained before)
                if (save_mask or save_npz) and mask is None:
                    seg_labels = camera.get_picture("Segmentation")
                    mask = seg_labels[..., 1].astype(np.uint8)
                    del seg_labels

                if save_mask and mask is not None:
                    _save_mask_image(mask, output_dirs['mask'], data_name)

                # Get depth data (if save_depth or save_npz is needed)
                depth = None
                if save_depth or save_npz:
                    camera_pos = camera.get_picture("Position")
                    depth = -camera_pos[..., 2]
                    del camera_pos
                    if save_depth:
                        _save_depth_data(depth, output_dirs['depth'], data_name)

                # Save npz data
                if save_npz:
                    intrinsic_matrix = camera.get_intrinsic_matrix()
                    extrinsic_matrix = get_camera_pos_mat(camera)
                    np.savez(os.path.join(output_dirs['npz'], f"{data_name}_data.npz"),
                             rgb=rgb,
                             part_mask=mask,
                             depth=depth,
                             intrinsic_matrix=intrinsic_matrix,
                             extrinsic_matrix=extrinsic_matrix)

            except Exception as e:
                tqdm.write(f"[Error] Error processing image {batch_idx + i}: {e}")
                continue

        pbar_cameras.close()

        # Explicitly delete scene and cameras to release GPU memory
        del scene, cameras
        gc.collect()

        # Check memory usage
        memory_percent = psutil.virtual_memory().percent
        pbar_batches.set_postfix({"Memory": f"{memory_percent:.1f}%"})

        # If memory is too high, pause for longer
        if memory_percent > 80:
            tqdm.write(f"[Warning] Memory usage too high ({memory_percent:.1f}%), "
                       f"pausing for 10 seconds...")
            time.sleep(10)

    pbar_batches.close()
    tqdm.write(f"[Info] All batches processed! Total cameras processed: {total_cameras}")


def render_multiview_images(urdf_name,
                            raw_data_dir,
                            phi_range,
                            theta_range,
                            phi_delta,
                            theta_delta,
                            cam_distance,
                            camera_width,
                            camera_height,
                            cover_mode,
                            save_rgb=True,
                            save_depth=False,
                            save_mask=False,
                            save_npz=True,
                            generate_colmap=False):
    """Render multi-view image data

    Args:
        urdf_name: URDF file name (without extension)
        raw_data_dir: Raw data directory
        phi_range: Elevation angle range (start, end)
        theta_range: Azimuth angle range (start, end)
        phi_delta: Elevation angle step size
        theta_delta: Azimuth angle step size
        cam_distance: Camera distance range (min, max)
        camera_width: Camera width
        camera_height: Camera height
        cover_mode: Whether to overwrite existing data
        save_rgb: Whether to save RGB images to images folder, default True
        save_depth: Whether to save depth maps to depths folder, default False
        save_mask: Whether to save mask images to masks folder, default False
        save_npz: Whether to save npz data to npzs folder, default True
        generate_colmap: Whether to generate COLMAP dataset. If True, automatically sets
                         save_mask=True, save_depth=True, save_npz=True

    Output folder structure:
        raw_data_dir/
            images/     (if save_rgb=True)
            depths/     (if save_depth=True)
            masks/      (if save_mask=True)
            npzs/       (if save_npz=True)
    """
    # If generating COLMAP dataset, automatically set required parameters
    if generate_colmap:
        save_mask = True
        save_depth = True
        save_npz = True
        tqdm.write("[Info] COLMAP dataset generation mode: automatically enabling "
                   "save_mask=True, save_depth=True, save_npz=True")

    if not any([save_rgb, save_depth, save_mask, save_npz]):
        raise ValueError("[Error] At least one output type must be selected "
                         "(save_rgb, save_depth, save_mask, save_npz)")

    # Handle cover mode: delete entire raw_data_dir and all its contents
    if cover_mode:
        if os.path.exists(raw_data_dir):
            print(f"[Warning] Old data exists, current mode is cover mode, "
                  f"deleting entire directory: {raw_data_dir}")
            shutil.rmtree(raw_data_dir)
        # Recreate empty directory
        os.makedirs(raw_data_dir, exist_ok=True)
    else:
        # Non-cover mode: check if output folders already exist
        os.makedirs(raw_data_dir, exist_ok=True)
        dir_mapping = {'images': save_rgb, 'depths': save_depth,
                       'masks': save_mask, 'npzs': save_npz}
        existing_dirs = [name for name, enabled in dir_mapping.items()
                        if enabled and os.path.exists(os.path.join(raw_data_dir, name))]
        if existing_dirs:
            raise ValueError(f"[Error] Old data exists, current mode is non-cover mode, "
                           f"please delete old data or use cover mode. Existing folders: {existing_dirs}")

    # Use batch processing
    process_in_batches(urdf_name, phi_range, theta_range, phi_delta, theta_delta,
                       cam_distance, camera_width, camera_height, raw_data_dir, cover_mode,
                       save_rgb, save_depth, save_mask, save_npz)

    # If generating COLMAP dataset, automatically call extract_camera_params
    if generate_colmap:
        from utils.camera_utils import extract_camera_params
        tqdm.write("[Info] Starting to generate COLMAP dataset...")
        npz_dir = os.path.join(raw_data_dir, "npzs")
        if not os.path.exists(npz_dir):
            raise ValueError(f"[Error] npz folder does not exist: {npz_dir}, "
                           f"cannot generate COLMAP dataset")
        extract_camera_params(
            file_path=npz_dir,
            save_path=raw_data_dir,
            height=camera_height,
            width=camera_width
        )
        tqdm.write(f"[Info] COLMAP dataset generated, saved at: "
                  f"{os.path.join(raw_data_dir, 'sparse', '0')}")

