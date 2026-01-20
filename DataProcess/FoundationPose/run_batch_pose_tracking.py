"""
Run batch pose tracking for FoundationPose:
- Traverse all record_* directories under --records_root
- Match organized_* / organzied_* (compatible with typo)
- Execute registration + tracking for each organized directory
- Output saved in <organized_dir>/tracking
"""

import logging
import trimesh
import imageio
import traceback
import numpy as np
import open3d as o3d
from pathlib import Path

print("[INFO] Precompiling nvdiffrast...")
try:
    import nvdiffrast.torch as dr
    # try to create a context to trigger compilation
    test_ctx = dr.RasterizeCudaContext()
    del test_ctx  # delete test context
    NVDIFFRAST_AVAILABLE = True
    print("[INFO] nvdiffrast compilation successful")
except Exception as e:
    print(f"[WARN] nvdiffrast compilation failed, will skip rasterization: {e}")
    dr = None
    NVDIFFRAST_AVAILABLE = False

from estimater import (
    ScorePredictor, PoseRefinePredictor, FoundationPose,
    set_logging_format, set_seed, draw_posed_3d_box, draw_xyz_axis
)
from datareader import (
    YcbineoatReader, depth2xyzmap, toOpen3dCloud
)


def find_organized_dirs(record_dir: Path):
    """
    Find organized data directories in a single record_xxx directory.
    Now organized folder is located inside cam_* folder.
    Compatible with 'organized_*' and typo 'organzied_*'.
    Also compatible with old structure (organized directly in record) and new structure (organized in cam_*).
    """
    cands = []
    
    # find organized in cam_* folder
    cam_dirs = sorted([d for d in record_dir.iterdir() if d.is_dir() and d.name.startswith("cam")])
    for cam_dir in cam_dirs:
        # find organized in cam_dir folder
        for pat in ["organized_*", "organzied_*"]:
            cands.extend(sorted(cam_dir.glob(pat)))
        # also compatible with 'organized' / 'organzied'
        for name in ["organized", "organzied"]:
            d = cam_dir / name
            if d.is_dir():
                cands.append(d)
    
    # old structure compatible: find organized in record folder (backward compatible)
    for pat in ["organized_*", "organzied_*"]:
        cands.extend(sorted(record_dir.glob(pat)))
    for name in ["organized", "organzied"]:
        d = record_dir / name
        if d.is_dir():
            cands.append(d)
    
    # Deduplicate
    uniq = []
    seen = set()
    for p in cands:
        if p.resolve() not in seen:
            uniq.append(p)
            seen.add(p.resolve())
    return uniq


def choose_mesh_file(organized_dir: Path, fallback_mesh: Path = None, idx: int=1 ) -> Path:
    """
    Prefer <organized_dir>/mesh/*.obj; otherwise use fallback_mesh specified on command line.
    """
    mesh_dir = organized_dir / f"mesh_{idx}"
    if mesh_dir.is_dir():
        objs = list(mesh_dir.glob("*.obj"))
        if len(objs) > 0:
            return objs[0]
    if fallback_mesh is not None:
        return fallback_mesh
    raise FileNotFoundError(f"no mesh .obj found: {mesh_dir} or --mesh_file not provided")


def ensure_empty_dir(p: Path, clean: bool):
    p.mkdir(parents=True, exist_ok=True)
    if clean:
        # Clean old content but keep root directory
        for sub in p.iterdir():
            if sub.is_file():
                sub.unlink()
            else:
                import shutil
                shutil.rmtree(sub, ignore_errors=True)


def run_one_organized(
    organized_dir: Path,
    mesh_file: Path = None,
    est_refine_iter: int = 10,
    track_refine_iter: int = 4,
    debug: int = 2,
    clean_output: bool = False,
    save_vis: bool = True,
    save_scene_pcd_on_first: bool = True,
    rgb_suffix: str = "png",
    index: int = 1
):
    """
    Perform full inference for a single organized directory.
    Returns:
        (ok: bool, err_msg: str)
    """
    try:
        tracking_name = f"tracking_{index}"
        organized_dir = organized_dir.resolve()
        out_dir = organized_dir / tracking_name
        if clean_output:
            logging.warning(f"[WARNING] about to clear old output directory: {out_dir}")
        ensure_empty_dir(out_dir, clean_output)
        (out_dir / "track_vis").mkdir(parents=True, exist_ok=True)
        (out_dir / "ob_in_cam").mkdir(parents=True, exist_ok=True)

        # mesh - load all mesh files and merge
        mesh_dir = organized_dir / f"mesh_{index}"
        mesh_list = []
        if mesh_dir.is_dir():
            mesh_files = sorted(list(mesh_dir.glob("*.obj")))
            if len(mesh_files) > 0:
                logging.info(f"[INFO] found {len(mesh_files)} mesh files, start loading and merging...")
                mesh_list = []
                mesh_file_mapping = []  # Track each mesh from which file
                
                for mesh_file_path in mesh_files:
                    loaded = trimesh.load(str(mesh_file_path))
                    # Process different mesh types
                    if isinstance(loaded, trimesh.Scene):
                        # If it's a scene, extract all geometries
                        for geom in loaded.geometry.values():
                            if isinstance(geom, trimesh.Trimesh):
                                mesh_list.append(geom)
                                mesh_file_mapping.append(mesh_file_path.name)
                    elif isinstance(loaded, trimesh.Trimesh):
                        mesh_list.append(loaded)
                        mesh_file_mapping.append(mesh_file_path.name)
                    else:
                        logging.warning(f"[WARN] skipping unsupported mesh type: {type(loaded)}")
                
                if len(mesh_list) == 0:
                    raise ValueError(f"no valid mesh object found in {mesh_dir}")
                
                # Check color information of each mesh before merging
                if len(mesh_list) > 1:
                    logging.info(f"[INFO] preparing to merge {len(mesh_list)} meshes, checking color information...")
                    for i, m in enumerate(mesh_list):
                        has_color = False
                        color_info = "no color"
                        if hasattr(m, 'visual') and m.visual is not None:
                            if isinstance(m.visual, trimesh.visual.ColorVisuals):
                                if m.visual.vertex_colors is not None and len(m.visual.vertex_colors) > 0:
                                    has_color = True
                                    colors = m.visual.vertex_colors
                                    color_info = f"vertex colors (shape: {colors.shape}, range: [{colors.min()}, {colors.max()}])"
                            elif isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
                                if m.visual.material is not None and m.visual.material.image is not None:
                                    has_color = True
                                    color_info = "texture mode"
                        source_file = mesh_file_mapping[i] if i < len(mesh_file_mapping) else "unknown"
                        if not has_color:
                            logging.warning(f"[WARN] Mesh {i} (from file: {source_file}) has no color information!")
                        else:
                            logging.info(f"[INFO] Mesh {i} (from file: {source_file}): {color_info}")
                
                # Merge all meshes
                if len(mesh_list) == 1:
                    mesh = mesh_list[0]
                    logging.info(f"[INFO] using single mesh, vertex count: {len(mesh.vertices)}")
                else:
                    mesh = trimesh.util.concatenate(mesh_list)
                    logging.info(f"[INFO] merged {len(mesh_list)} meshes, total vertex count: {len(mesh.vertices)}")
                    
                    # Check color information is preserved after merging
                    if hasattr(mesh, 'visual') and mesh.visual is not None:
                        if isinstance(mesh.visual, trimesh.visual.ColorVisuals):
                            if mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
                                colors = mesh.visual.vertex_colors
                                logging.info(f"[INFO] merged mesh retains vertex colors, shape: {colors.shape}, range: [{colors.min()}, {colors.max()}]")
                                if colors.shape[0] > 0:
                                    unique_colors = np.unique(colors.reshape(-1, 3), axis=0)
                                    if len(unique_colors) == 1 and np.allclose(unique_colors[0], [128, 128, 128]):
                                        logging.warning(f"[WARN] all vertex colors are gray(128,128,128) after merging, color information may not be correctly preserved!")
                                    else:
                                        logging.info(f"[INFO] merged mesh has {len(unique_colors)} different colors")
                            else:
                                logging.error(f"[ERROR] merged mesh's vertex_colors is None or empty! color information lost!")
                        elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
                            if mesh.visual.material is not None and mesh.visual.material.image is not None:
                                logging.info(f"[INFO] merged mesh uses texture mode")
                            else:
                                logging.error(f"[ERROR] merged mesh texture missing!")
                        else:
                            logging.warning(f"[WARN] merged mesh's visual type unknown: {type(mesh.visual)}")
                    else:
                        logging.error(f"[ERROR] merged mesh has no visual attribute! color information lost!")
            else:
                # If no mesh file found, try using fallback
                if mesh_file is not None:
                    logging.info(f"[INFO] no mesh file found in {mesh_dir}, using fallback: {mesh_file}")
                    mesh = trimesh.load(str(mesh_file))
                    if isinstance(mesh, trimesh.Scene):
                        mesh_list = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                        if len(mesh_list) > 0:
                            # Check color information of each mesh
                            if len(mesh_list) > 1:
                                logging.info(f"[INFO] Fallback mesh has {len(mesh_list)} submeshes, checking color information...")
                                for i, m in enumerate(mesh_list):
                                    has_color = False
                                    if hasattr(m, 'visual') and m.visual is not None:
                                        if isinstance(m.visual, trimesh.visual.ColorVisuals):
                                            has_color = m.visual.vertex_colors is not None and len(m.visual.vertex_colors) > 0
                                        elif isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
                                            has_color = m.visual.material is not None and m.visual.material.image is not None
                                    if not has_color:
                                        logging.warning(f"[WARN] Fallback mesh submesh {i} has no color information")
                            mesh = trimesh.util.concatenate(mesh_list) if len(mesh_list) > 1 else mesh_list[0]
                            # Check color information is preserved after merging
                            if hasattr(mesh, 'visual') and mesh.visual is not None:
                                if isinstance(mesh.visual, trimesh.visual.ColorVisuals):
                                    if mesh.visual.vertex_colors is not None:
                                        logging.info(f"[INFO] Fallback mesh retains vertex colors after merging")
                                    else:
                                        logging.error(f"[ERROR] Fallback mesh loses vertex colors after merging")
                    elif not isinstance(mesh, trimesh.Trimesh):
                        raise ValueError(f"Unexpected mesh type: {type(mesh)}")
                else:
                    raise FileNotFoundError(f"no mesh .obj found: {mesh_dir} or --mesh_file not provided")
        else:
            # If mesh_dir does not exist, use fallback
            if mesh_file is not None:
                logging.info(f"[INFO] mesh_dir does not exist, using fallback: {mesh_file}")
                mesh = trimesh.load(str(mesh_file))
                if isinstance(mesh, trimesh.Scene):
                    mesh_list = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                    if len(mesh_list) > 0:
                        # Check color information of each mesh
                        if len(mesh_list) > 1:
                            logging.info(f"[INFO] Fallback mesh has {len(mesh_list)} submeshes, checking color information...")
                            for i, m in enumerate(mesh_list):
                                has_color = False
                                if hasattr(m, 'visual') and m.visual is not None:
                                    if isinstance(m.visual, trimesh.visual.ColorVisuals):
                                        has_color = m.visual.vertex_colors is not None and len(m.visual.vertex_colors) > 0
                                    elif isinstance(m.visual, trimesh.visual.texture.TextureVisuals):
                                        has_color = m.visual.material is not None and m.visual.material.image is not None
                                if not has_color:
                                    logging.warning(f"[WARN] Fallback mesh submesh {i} has no color information")
                        mesh = trimesh.util.concatenate(mesh_list) if len(mesh_list) > 1 else mesh_list[0]
                        # Check color information is preserved after merging
                        if hasattr(mesh, 'visual') and mesh.visual is not None:
                            if isinstance(mesh.visual, trimesh.visual.ColorVisuals):
                                if mesh.visual.vertex_colors is not None:
                                    logging.info(f"[INFO] Fallback mesh retains vertex colors after merging")
                                else:
                                    logging.error(f"[ERROR] Fallback mesh loses vertex colors after merging")
                elif not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(f"Unexpected mesh type: {type(mesh)}")
            else:
                raise FileNotFoundError(f"no mesh directory found: {mesh_dir} or --mesh_file not provided")

        # Ensure mesh has vertices attribute
        if not hasattr(mesh, 'vertices'):
            raise ValueError(f"Mesh object does not have vertices attribute: {type(mesh)}")

        # Process texture and color information, ensure color information is correctly loaded
        try:
            # Check if there is texture information
            if (hasattr(mesh, 'visual') and mesh.visual is not None and
                isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals)):
                # Check if texture is valid
                if (mesh.visual.material is None or 
                    mesh.visual.material.image is None):
                    logging.warning(f"[WARN] Mesh texture missing, trying to extract color from MTL material...")
                    # Try to find color information from other attributes of mesh
                    vertex_colors_found = None
                    
                    # Method 1: Check if there is vertex color data (some formats may contain both texture and vertex colors)
                    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                        vertex_colors_found = mesh.visual.vertex_colors
                        logging.info(f"[INFO] found vertex colors in TextureVisuals, converting to vertex color mode")
                    # Method 2: Check if mesh has other color attributes
                    elif hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None:
                        vertex_colors_found = mesh.vertex_colors
                        logging.info(f"[INFO] found vertex colors in mesh attributes")
                    # Method 3: Extract Kd (diffuse color) value from MTL material
                    elif mesh.visual.material is not None:
                        # Try to get Kd color from material
                        material = mesh.visual.material
                        kd_color = None
                        
                        # Check if there is main_color attribute (trimesh may store Kd here)
                        if hasattr(material, 'main_color') and material.main_color is not None:
                            kd_color = material.main_color
                            logging.info(f"[INFO] extracted color from material.main_color: {kd_color}")
                        # Check if there is diffuse attribute
                        elif hasattr(material, 'diffuse') and material.diffuse is not None:
                            kd_color = material.diffuse
                            logging.info(f"[INFO] extracted color from material.diffuse: {kd_color}")
                        # Check if there is Kd attribute (some versions may store it directly as Kd)
                        elif hasattr(material, 'Kd') and material.Kd is not None:
                            kd_color = material.Kd
                            logging.info(f"[INFO] extracted color from material.Kd: {kd_color}")
                        # Check if there is baseColorFactor attribute (glTF format)
                        elif hasattr(material, 'baseColorFactor') and material.baseColorFactor is not None:
                            kd_color = material.baseColorFactor
                            logging.info(f"[INFO] extracted color from material.baseColorFactor: {kd_color}")
                        
                        if kd_color is not None:
                            # Convert Kd color to 0-255 range RGB
                            kd_color_array = np.array(kd_color)
                            logging.info(f"[DEBUG] extracted kd_color original value: {kd_color_array}, length: {len(kd_color_array)}")
                            if len(kd_color_array) >= 3:
                                kd_rgb = kd_color_array[:3]
                                logging.info(f"[DEBUG] kd_rgb (first 3 values): {kd_rgb}, max value: {kd_rgb.max()}")
                                # If color value is in 0-1 range, convert to 0-255
                                if kd_rgb.max() <= 1.0:
                                    kd_color_rgb = (kd_rgb * 255).astype(np.uint8)
                                    logging.info(f"[DEBUG] color value in 0-1 range, converted: {kd_color_rgb}")
                                else:
                                    # If already in 0-255 range, use directly
                                    kd_color_rgb = kd_rgb.astype(np.uint8)
                                    logging.info(f"[DEBUG] color value in 0-255 range, use directly: {kd_color_rgb}")
                                
                                # Apply same color to all vertices
                                vertex_colors_found = np.tile(
                                    kd_color_rgb.reshape(1, 3),
                                    (len(mesh.vertices), 1)
                                )
                                logging.info(f"[INFO] extracted color from MTL material and applied to all vertices: RGB={kd_color_rgb}, vertex count={len(mesh.vertices)}")
                                logging.info(f"[DEBUG] vertex_colors_found shape: {vertex_colors_found.shape}, example: {vertex_colors_found[:3]}")
                    
                    if vertex_colors_found is not None and len(vertex_colors_found) > 0:
                        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors_found)
                        logging.info(f"[INFO] successfully converted to vertex color mode")
                    else:
                        logging.error(f"[ERROR] Mesh texture missing and cannot extract color information!")
                        logging.error(f"[ERROR] diagnostic information:")
                        logging.error(f"[ERROR]   - visual type: {type(mesh.visual)}")
                        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                            material = mesh.visual.material
                            logging.error(f"[ERROR]   - material type: {type(material)}")
                            logging.error(f"[ERROR]   - material attributes: {[attr for attr in dir(material) if not attr.startswith('_')]}")
                            # Try to print all possible color attributes of material
                            for attr in ['main_color', 'diffuse', 'Kd', 'ambient', 'specular', 'baseColorFactor']:
                                if hasattr(material, attr):
                                    try:
                                        value = getattr(material, attr)
                                        logging.error(f"[ERROR]   - material.{attr}: {value}")
                                    except Exception as e:
                                        logging.error(f"[ERROR]   - material.{attr}: cannot access ({e})")
                        else:
                            logging.error(f"[ERROR]   - has material: N/A")
                        logging.error(f"[ERROR]   - vertex count: {len(mesh.vertices)}")
                        logging.error(f"[ERROR] suggestions: please check if mesh file (.obj) contains:")
                        logging.error(f"[ERROR]   1. texture file (.mtl + image file)")
                        logging.error(f"[ERROR]   2. vertex color information (vc/vn etc.)")
                        logging.error(f"[ERROR]   3. Kd color value in MTL file")
                        raise ValueError("Mesh has no valid texture and vertex colors, and cannot extract color from MTL material, cannot perform color-based pose tracking.")
                else:
                    logging.info(f"[INFO] Mesh uses texture mode, texture size: {mesh.visual.material.image.size}")
            elif (hasattr(mesh, 'visual') and mesh.visual is not None and
                  isinstance(mesh.visual, trimesh.visual.ColorVisuals)):
                # Already in vertex color mode, check if there is color information
                if mesh.visual.vertex_colors is None or len(mesh.visual.vertex_colors) == 0:
                    logging.error(f"[ERROR] Mesh vertex colors are None or empty! This will make all objects look the same, and cannot distinguish objects with different colors!")
                    raise ValueError("Mesh vertex colors are missing, cannot perform color-based tracking. Please check if mesh file contains vertex color information.")
                else:
                    # Verify if color values are reasonable
                    colors = mesh.visual.vertex_colors
                    # Check if all colors are gray (may indicate color information not correctly loaded)
                    if colors.shape[0] > 0:
                        unique_colors = np.unique(colors.reshape(-1, 3), axis=0)
                        if len(unique_colors) == 1:
                            if np.allclose(unique_colors[0], [128, 128, 128]):
                                logging.error(f"[ERROR] All vertex colors are gray(128,128,128)! Color information may not be correctly loaded! This will cause tracking to fail to distinguish objects with different colors!")
                                raise ValueError("All vertex colors are gray, color information may not be correctly loaded. Please check if mesh file contains correct color information.")
                            else:
                                logging.warning(f"[WARN] All vertex colors are the same: {unique_colors[0]}, object may be single-colored")
                        logging.info(f"[INFO] Mesh uses vertex color mode, vertex count: {len(colors)}, color range: [{colors.min()}, {colors.max()}], unique color count: {len(unique_colors)}")
            else:
                # No visual information
                logging.error(f"[ERROR] Mesh has no visual attribute or visual is None!")
                logging.error(f"[ERROR] diagnostic information:")
                logging.error(f"[ERROR]   - has visual attribute: {hasattr(mesh, 'visual')}")
                if hasattr(mesh, 'visual'):
                    logging.error(f"[ERROR]   - visual value: {mesh.visual}")
                logging.error(f"[ERROR]   - vertex count: {len(mesh.vertices)}")
                logging.error(f"[ERROR]   - face count: {len(mesh.faces)}")
                logging.error(f"[ERROR] suggestions: please check if mesh file (.obj) contains:")
                logging.error(f"[ERROR]   1. texture file (.mtl + image file) - need usemtl and mtllib instructions")
                logging.error(f"[ERROR]   2. vertex color information - use 'v x y z r g b' format in .obj file")
                logging.error(f"[ERROR]   3. or use trimesh to add colors to mesh: mesh.visual.vertex_colors = colors")
                raise ValueError("Mesh has no visual information (texture or vertex colors), cannot perform color-based pose tracking. Please check if mesh file contains color or texture information.")
        except ValueError:
            # Re-throw ValueError, no fallback handling
            raise
        except Exception as e:
            logging.error(f"[ERROR] texture/color processing failed: {e}")
            logging.error(f"[ERROR] cannot continue, because color information is crucial for distinguishing different objects!")
            raise ValueError(f"Mesh color information processing failed: {e}. Cannot perform color-based pose tracking.")

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        # model / predictor
        # if dr is None:
        #     raise RuntimeError("nvdiffrast is not ready, please install nvdiffrast and ensure it can be imported.")
        # glctx = dr.RasterizeCudaContext()
        if NVDIFFRAST_AVAILABLE:
            glctx = dr.RasterizeCudaContext()
        else:
            glctx = None
            logging.warning("nvdiffrast is not available, will skip rasterization function")

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=str(out_dir),
            debug=debug,
            glctx=glctx
        )
        logging.info(f"[INIT] Estimator initialized | organized={organized_dir.name}")

        # data reader (same as your demo)
        reader = YcbineoatReader(
            video_dir=str(F"{organized_dir}"),
            shorter_side=None,
            zfar=np.inf,
            suffix=rgb_suffix
        )

        pose = None
        for i in range(len(reader.color_files)):
            logging.info(f"[{organized_dir.name}] frame {i+1}/{len(reader.color_files)}")
            color = reader.get_color(i)
            depth = reader.get_depth(i)

            if i == 0:
                # Only first frame needs mask
                mask = reader.get_mask(i=0,idx=index).astype(bool)
                pose = est.register(
                    K=reader.K,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=est_refine_iter
                )
                if debug >= 3:
                    m = mesh.copy()
                    m.apply_transform(pose)
                    m.export(str(out_dir / "model_tf.obj"))

                if save_scene_pcd_on_first and debug >= 1:
                    xyz_map = depth2xyzmap(depth, reader.K)
                    valid = depth >= 0.001
                    pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                    o3d.io.write_point_cloud(str(out_dir / "scene_complete.ply"), pcd)
            else:
                pose = est.track_one(
                    rgb=color,
                    depth=depth,
                    K=reader.K,
                    iteration=track_refine_iter
                )

            # Save pose (object in camera coordinate system)
            np.savetxt(out_dir / "ob_in_cam" / f"{reader.id_strs[i]}.txt", pose.reshape(4, 4))

            # Visualization (no popup, only save)
            if debug >= 1 and save_vis:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                imageio.imwrite(out_dir / "track_vis" / f"{reader.id_strs[i]}.png", vis)

        logging.info(f"[DONE] {organized_dir.name} output directory: {out_dir}")
        return True, ""
    except Exception as e:
        # Record stack, return error message
        logging.exception(f"[ERROR] processing {organized_dir} failed: {e}")
        return False, traceback.format_exc(limit=3)


def batch_pose_tracking(
    idx = None,
    records_root = None,
    mesh_file = None,      # can be omitted, default using the first obj file in organized_dir/mesh_idx/*.obj
    est_refine_iter = 10,  # registration iterations
    track_refine_iter = 4, # tracking iterations
    debug = 3,              # debug level, 0-no output, 1-output key information, 2-output key information+debug information, 3-output debug information
    rgb_suffix = "png",     # rgb image suffix, "jpg" or "png"
    clean_output = True,    # clean each organized/tracking_idx content before processing
    seed = 0,               # random seed
    ):                
    assert idx is not None, "idx cannot be empty"
    assert records_root is not None, "records_root cannot be empty"
    records_root = Path(records_root).resolve()
    # Collect all record_* directories
    record_dirs = sorted([p for p in records_root.glob("record_*") if p.is_dir()])
    if len(record_dirs) == 0:
        logging.warning(f"No record_* directories found in {records_root}")
        return

    logging.info(f"Found {len(record_dirs)} record directories. Starting batch processing...")

    fallback_mesh = Path(mesh_file).resolve() if mesh_file else None

    total_orgs = 0
    succ_cnt, fail_cnt, skip_cnt = 0, 0, 0
    succ_list, skip_list = [], []
    fail_items = []  # [(organized_path, err_msg)]

    for rec in record_dirs:
        logging.info(f"[DEBUG] processing record: {rec.name}")
        org_dirs = find_organized_dirs(rec)
        if len(org_dirs) == 0:
            logging.warning(f"[SKIP] no organized data directories found in {rec.name}")
            skip_cnt += 1
            skip_list.append(str(rec))
            continue

        for org in org_dirs:
            logging.info(f"[DEBUG] processing organized: {org.name}")
            total_orgs += 1
            ok, err = run_one_organized(
                organized_dir=org,
                mesh_file=fallback_mesh,
                est_refine_iter=est_refine_iter,    # registration iterations
                track_refine_iter=track_refine_iter, # tracking iterations
                debug=debug,
                clean_output=clean_output,              # clean each organized/tracking_idx content before processing
                save_vis=True,
                save_scene_pcd_on_first=True,          # save scene point cloud (but only save when debug>=1)
                rgb_suffix=rgb_suffix,                  # rgb image suffix, "jpg" or "png"
                index=idx                               # object idx
            )
            if ok:
                succ_cnt += 1
                succ_list.append(str(org))
                logging.info(f"[DEBUG] success: {org.name}")
            else:
                fail_cnt += 1
                fail_items.append((str(org), err.strip()))
                logging.error(f"[DEBUG] failure details: {err}")

    # ===== Summary =====
    logging.info("=" * 70)
    logging.info(f"[SUMMARY] total organized directories: {total_orgs}")
    logging.info(f"[SUMMARY] success: {succ_cnt}  |  failure: {fail_cnt}  |  skipped (no organized_*): {skip_cnt}")
    if fail_cnt > 0:
        logging.warning("[FAILURE LIST] ↓")
        for p, _ in fail_items:
            logging.warning(f" - {p}")
    if succ_cnt > 0:
        logging.info("[SUCCESS LIST] ↓")
        for p in succ_list:
            logging.info(f" - {p}")
    if skip_cnt > 0:
        logging.info("[SKIPPED LIST] ↓")
        for p in skip_list:
            logging.info(f" - {p}")

    # Write summary file
    summary_txt = records_root / "batch_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Total organized: {total_orgs}\nSuccess: {succ_cnt}\nFailure: {fail_cnt}\nSkipped: {skip_cnt}\n")
        if fail_cnt:
            f.write("\n[FAILURE LIST]\n")
            for p, err in fail_items:
                f.write(f"- {p}\n")
                # Only write first line of error and a small traceback, avoid file too long
                first_line = err.splitlines()[0] if err else ""
                f.write(f"  error: {first_line}\n")
        if succ_cnt:
            f.write("\n[SUCCESS LIST]\n" + "\n".join(succ_list) + "\n")
        if skip_cnt:
            f.write("\n[SKIPPED LIST]\n" + "\n".join(skip_list) + "\n")
    logging.info(f"[SUMMARY] written to: {summary_txt}")

