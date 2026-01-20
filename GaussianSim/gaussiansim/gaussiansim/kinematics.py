import os
import torch
import trimesh
import numpy as np
from urdfpy import URDF
import matplotlib.pyplot as plt

from .model import GaussianModel

class URDFKinematics:
    """
    A class to load a URDF, set joint states, and compute forward kinematics for all links.
    """

    def __init__(self, urdf_path, joint_states=None, T_root=None):
        """
        Initialize the URDFKinematics object.

        :param urdf_path: Path to the URDF file.
        :param joint_states: Optional dict { joint_name: value } where value is joint angle (rad) or translation (m).
                             Any missing joint will default to 0.0.
        :param T_root: Optional 4×4 numpy.ndarray specifying the root link's transform in the world frame.
                       If None, defaults to identity.
        """
        # Load URDF
        self.robot = URDF.load(urdf_path)
        self.urdf_dir = os.path.dirname(urdf_path)

        # Determine root link name
        self.root_link = self.robot.base_link.name

        # Initialize joint states (default to 0.0 for any missing joint)
        if joint_states is None:
            joint_states = {}
        for joint in self.robot.joints:
            if joint.name not in joint_states:
                joint_states[joint.name] = 0.0
        self.joint_states = joint_states

        # Root transform in world frame (identity if not provided)
        self.T_root = np.eye(4) if T_root is None else T_root

        # Build parent_link → [joint, ...] mapping for quick traversal
        self.parent_map = {}
        for joint in self.robot.joints:
            parent = joint.parent
            self.parent_map.setdefault(parent, []).append(joint)

        # Placeholder for computed link poses
        self.link_poses = {}
        self.compute_link_poses()  # Compute initial poses

    @staticmethod
    def rpy_to_rot(rpy):
        """
        Convert (roll, pitch, yaw) Euler angles into a 3×3 rotation matrix.
        Euler order: rotate about x (roll), then y (pitch), then z (yaw).
        """
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Rotation about x-axis
        R_x = np.array([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr,  cr]
        ])
        # Rotation about y-axis
        R_y = np.array([
            [cp, 0, sp],
            [0, 1,  0],
            [-sp,0, cp]
        ])
        # Rotation about z-axis
        R_z = np.array([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [0,   0,  1]
        ])
        # Combined rotation: R = R_z @ R_y @ R_x
        return R_z @ R_y @ R_x

    @staticmethod
    def transform_from_origin(xyz, rpy):
        """
        Construct a 4×4 homogeneous transform: first rotation, then translation.
        :param xyz: [x, y, z]
        :param rpy: [roll, pitch, yaw]
        """
        R = URDFKinematics.rpy_to_rot(rpy)
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3]   = np.array(xyz).reshape(3)
        return T

    @staticmethod
    def transform_from_axis_rotation(axis, theta):
        """
        Compute a 4×4 homogeneous transform for rotation about a given unit axis by angle theta (radians).
        Uses the Rodrigues formula.
        """
        axis = np.array(axis, dtype=float)
        if np.linalg.norm(axis) < 1e-8:
            return np.eye(4)
        u = axis / np.linalg.norm(axis)
        ux, uy, uz = u

        c = np.cos(theta)
        s = np.sin(theta)
        one_c = 1 - c

        # Skew-symmetric matrix [u]_x
        Ux = np.array([
            [   0, -uz,  uy],
            [  uz,   0, -ux],
            [ -uy,  ux,   0]
        ])
        # Outer product u u^T
        uuT = np.outer(u, u)
        R = c * np.eye(3) + one_c * uuT + s * Ux

        T = np.eye(4)
        T[0:3, 0:3] = R
        return T

    @staticmethod
    def transform_from_axis_translation(axis, d):
        """
        Compute a 4×4 homogeneous transform for translation of distance d along the given unit axis.
        """
        axis = np.array(axis, dtype=float)
        if np.linalg.norm(axis) < 1e-8 or abs(d) < 1e-12:
            return np.eye(4)
        u = axis / np.linalg.norm(axis)
        T = np.eye(4)
        T[0:3, 3] = u * d
        return T

    def _recurse_links(self, current_link, T_parent):
        """
        Private recursive function to traverse from current_link downward and compute each link's transform.
        :param current_link: Name of the current link being processed.
        :param T_parent: 4×4 numpy.ndarray representing current_link's transform in the world frame.
        """
        # Record the current link's pose
        self.link_poses[current_link] = T_parent

        # If there are no child joints, return
        if current_link not in self.parent_map:
            return

        # Iterate over each joint whose parent is current_link
        for joint in self.parent_map[current_link]:
            # 1. Get the static offset T_origin
            #    In newer urdfpy versions, joint.origin may already be a 4×4 ndarray
            if isinstance(joint.origin, np.ndarray):
                T_origin = joint.origin
            else:
                # if joint.origin is the old object with xyz, rpy attributes, decompose compatibly
                xyz = joint.origin.xyz.tolist()
                rpy = joint.origin.rpy.tolist()
                T_origin = URDFKinematics.transform_from_origin(xyz, rpy)

            # 2. Build the motion transform T_motion based on joint type & state
            theta_or_d = self.joint_states[joint.name]
            if joint.joint_type in ['revolute', 'continuous']:
                axis = joint.axis.tolist()
                T_motion = URDFKinematics.transform_from_axis_rotation(axis, theta_or_d)
            elif joint.joint_type == 'prismatic':
                axis = joint.axis.tolist()
                T_motion = URDFKinematics.transform_from_axis_translation(axis, theta_or_d)
            else:
                # fixed, floating, planar, etc. → identity
                T_motion = np.eye(4)

            # 3. Compute child link's global transform: T_child = T_parent @ T_origin @ T_motion
            T_child = T_parent @ T_origin @ T_motion

            # 4. Recurse into the child link
            self._recurse_links(joint.child, T_child)

    def compute_link_poses(self):
        """
        Compute forward kinematics for all links, populating self.link_poses.
        :return: Dictionary { link_name: 4×4 numpy.ndarray } representing each link's transform in the world frame.
        """
        # Clear any previous results
        self.link_poses.clear()

        # Start recursion from the root link, whose transform is self.T_root
        self._recurse_links(self.root_link, self.T_root)

    
    def _set_equal_aspect(self, ax, origins, axis_length):
        """
        Force equal scaling on a Matplotlib 3D Axes so X, Y, Z axes appear equal.
        :param ax: a 3D axes instance (Axes3D)
        :param origins: an (N×3) array of all link origin coords
        :param axis_length: scalar for arrow length padding
        """
        # Compute min/max over x,y,z, then pad by axis_length
        mins = origins.min(axis=0) - axis_length
        maxs = origins.max(axis=0) + axis_length
        max_range = (maxs - mins).max()

        # Center
        center = (mins + maxs) / 2.0
        half = max_range / 2.0

        # Set each axis to span [center-half, center+half]
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)


    def visualize_all_links(self, axis_length=0.05):
        """
        Visualize all links' poses as 3D coordinate frames in Matplotlib,
        and draw lines from each parent link origin to its child link origin.

        :param axis_length: Length of each coordinate arrow (in meters).
        """
        if not self.link_poses:
            raise RuntimeError("No link poses computed. Call compute_link_poses() first.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("All Link Poses in World Frame with Parent-Child Connections")

        origins = []

        # 1) Draw each link's coordinate frame
        for link_name, T in self.link_poses.items():
            origin = T[0:3, 3]
            origins.append(origin)
            R = T[0:3, 0:3]

            # Compute x,y,z unit vectors in world from R
            x_dir = (R @ np.array([1.0, 0.0, 0.0])) * axis_length
            y_dir = (R @ np.array([0.0, 1.0, 0.0])) * axis_length
            z_dir = (R @ np.array([0.0, 0.0, 1.0])) * axis_length

            # Plot the origin point
            ax.scatter(origin[0], origin[1], origin[2], color='k', s=10)

            # Plot the three axes as quivers
            ax.quiver(
                origin[0], origin[1], origin[2],
                x_dir[0], x_dir[1], x_dir[2],
                color='r', length=axis_length, normalize=False
            )
            ax.quiver(
                origin[0], origin[1], origin[2],
                y_dir[0], y_dir[1], y_dir[2],
                color='g', length=axis_length, normalize=False
            )
            ax.quiver(
                origin[0], origin[1], origin[2],
                z_dir[0], z_dir[1], z_dir[2],
                color='b', length=axis_length, normalize=False
            )

            # Label with link name
            ax.text(origin[0], origin[1], origin[2], f"{link_name}", size=6, zorder=1)

        origins = np.array(origins)

        # 2) Draw lines from each parent origin to child origin
        for joint in self.robot.joints:
            parent_name = joint.parent
            child_name = joint.child
            if parent_name in self.link_poses and child_name in self.link_poses:
                p_o = self.link_poses[parent_name][0:3, 3]
                c_o = self.link_poses[child_name][0:3, 3]
                ax.plot(
                    [p_o[0], c_o[0]],
                    [p_o[1], c_o[1]],
                    [p_o[2], c_o[2]],
                    color='gray', linestyle='-', linewidth=1
                )

        # 3) Enforce truly equal aspect ratio
        self._set_equal_aspect(ax, origins, axis_length)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def visualize_full_robot(self, view_link_pose=False, view_robot=True):
        """
        Render the entire URDF in its current joint configuration.
        If urdfpy.scene() exists, use it. Otherwise, manually load/generate each geometry,
        ensuring we convert any loaded trimesh.Scene into a Trimesh before applying transforms.
        """
        # Attempt to use urdfpy's scene() if available
        if hasattr(self.robot, 'scene'):
            scene = self.robot.scene(joint_angles=self.joint_states)
            scene.show()
            return

        # Fallback: manually assemble a trimesh.Scene by loading or generating each geometry
        scene = trimesh.Scene()
        link_trans_poses = {}

        for link_name, T_link in self.link_poses.items():
            link_obj = self.robot.link_map[link_name]
            for visual in link_obj.visuals:
                geom = visual.geometry
                mesh = None
                # 1) If a mesh is specified, load it from file
                if geom.mesh is not None:
                    mesh_info = geom.mesh  # URDFpy's Mesh object
                    filename = mesh_info.filename
                    # pdb.set_trace()  # For debugging, can be removed in production code
                    # Resolve relative paths against URDF directory
                    if not os.path.isabs(filename):
                        filename = os.path.join(self.urdf_dir, filename)
                    loaded = trimesh.load(filename, force='mesh')

                    # If load returned a Scene, combine into one Trimesh
                    if isinstance(loaded, trimesh.Scene):
                        mesh = loaded.dump(concatenate=True)
                    else:
                        mesh = loaded

                    # Apply scaling if provided
                    if mesh_info.scale is not None:
                        sx, sy, sz = mesh_info.scale
                        scale_matrix = np.diag([sx, sy, sz, 1.0])
                        mesh.apply_transform(scale_matrix)

                # 2) Otherwise, if a box is specified, create it
                elif geom.box is not None:
                    size = geom.box.size  # [x, y, z]
                    mesh = trimesh.creation.box(extents=size)

                # 3) Otherwise, if a cylinder is specified, create it
                elif geom.cylinder is not None:
                    radius = geom.cylinder.radius
                    length = geom.cylinder.length
                    mesh = trimesh.creation.cylinder(radius=radius, height=length)

                # 4) Otherwise, if a sphere is specified, create it
                elif geom.sphere is not None:
                    radius = geom.sphere.radius
                    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)

                else:
                    # No known geometry type; skip
                    continue

                # Build the visual-origin transform (4×4)
                if isinstance(visual.origin, np.ndarray):
                    T_vis = visual.origin
                else:
                    T_vis = URDFKinematics.transform_from_origin(
                        visual.origin.xyz.tolist(),
                        visual.origin.rpy.tolist()
                    )

                # Apply transforms: first the link’s global pose, then the visual origin
                mesh.apply_transform(T_link @ T_vis)
                # link_trans_poses[link_name] = T_link @ T_vis
                link_trans_poses[link_name] = T_link
                # if not np.array_equal(T_vis, np.eye(4)):
                    # print(f"Link <{link_name}> pose in world frame:\n{T_link}\n")
                    # print(f"Visual origin for link <{link_name}>:\n{T_vis}\n")
                    # Add to the scene
                scene.add_geometry(mesh)
                if view_link_pose:
                    # Add a coordinate frame for this link
                    axes = trimesh.creation.axis(
                        origin_size=0.01,
                        axis_length=0.2,
                        axis_radius=0.005)
                    axes.apply_transform(T_link @ T_vis)
                    scene.add_geometry(axes)
        
        # Add world coordinate frame
        world_axes = trimesh.creation.axis(
            origin_size=0.001,
            axis_length=0.2,
            axis_radius=0.005
        )
        scene.add_geometry(world_axes)
        
        # Show the assembled scene
        if view_robot:
            scene.show()
        return link_trans_poses
    

    def get_new_link_tran_poses(self, joint_states, view_tansform=False):
        """Compute link poses for given joint states and optionally visualize."""
        # Initialize joint states (default to 0.0 for any missing joint)
        if joint_states is None:
            joint_states = {}
        for joint in self.robot.joints:
            if joint.name not in joint_states:
                joint_states[joint.name] = 0.0
        self.joint_states = joint_states
        self.compute_link_poses()
        link_trans_poses = self.visualize_full_robot(view_link_pose=False, view_robot=view_tansform)
        return link_trans_poses

    def _recurse_links_gpu(self, current_link, T_parent, device='cuda'):
        """GPU version of recursive link traversal for forward kinematics."""
        self.link_poses_gpu[current_link] = T_parent.clone()
        if current_link not in self.parent_map:
            return
        for joint in self.parent_map[current_link]:
            if isinstance(joint.origin, np.ndarray):
                T_origin = torch.from_numpy(joint.origin).float().to(device)
            else:
                T_origin_np = URDFKinematics.transform_from_origin(joint.origin.xyz.tolist(), joint.origin.rpy.tolist())
                T_origin = torch.from_numpy(T_origin_np).float().to(device)
            theta_or_d = self.joint_states[joint.name]
            if joint.joint_type in ['revolute', 'continuous']:
                T_motion_np = URDFKinematics.transform_from_axis_rotation(joint.axis.tolist(), theta_or_d)
                T_motion = torch.from_numpy(T_motion_np).float().to(device)
            elif joint.joint_type == 'prismatic':
                T_motion_np = URDFKinematics.transform_from_axis_translation(joint.axis.tolist(), theta_or_d)
                T_motion = torch.from_numpy(T_motion_np).float().to(device)
            else:
                T_motion = torch.eye(4, device=device)
            T_child = T_parent @ T_origin @ T_motion
            self._recurse_links_gpu(joint.child, T_child, device)
    
    def compute_link_poses_gpu(self, device='cuda'):
        """Compute link poses on GPU for faster computation."""
        if not hasattr(self, 'link_poses_gpu'):
            self.link_poses_gpu = {}
        else:
            self.link_poses_gpu.clear()
        T_root_gpu = torch.from_numpy(self.T_root).float().to(device)
        self._recurse_links_gpu(self.root_link, T_root_gpu, device)
    
    def get_link_transforms_only(self):
        """Return dictionary of link transforms without visualization."""
        return {link_name: T_link for link_name, T_link in self.link_poses.items()}
    
    def get_new_link_tran_poses_gpu(self, joint_states, device='cuda', keep_on_gpu=True):
        """Compute link poses on GPU for given joint states."""
        if joint_states is None:
            joint_states = {}
        for joint in self.robot.joints:
            if joint.name not in joint_states:
                joint_states[joint.name] = 0.0
        self.joint_states = joint_states
        self.compute_link_poses_gpu(device=device)
        link_trans_poses = {}
        for link_name, T_link in self.link_poses_gpu.items():
            link_trans_poses[link_name] = T_link if keep_on_gpu else T_link.cpu().numpy()
        return link_trans_poses
    
def build_rotation_tensor(q):
    """Convert quaternion tensor to rotation matrix tensor."""
    norm = torch.sqrt(q[:, 0]**2 + q[:, 1]**2 + q[:, 2]**2 + q[:, 3]**2)
    q = q / norm[:, None]
    
    rot = torch.zeros((q.size(0), 3, 3), device=q.device)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    rot[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rot[:, 0, 1] = 2 * (x*y - r*z)
    rot[:, 0, 2] = 2 * (x*z + r*y)
    rot[:, 1, 0] = 2 * (x*y + r*z)
    rot[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rot[:, 1, 2] = 2 * (y*z - r*x)
    rot[:, 2, 0] = 2 * (x*z - r*y)
    rot[:, 2, 1] = 2 * (y*z + r*x)
    rot[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return rot

def rotmat_to_quaternion_tensor(R):
    """Convert rotation matrix tensor to quaternion tensor using Shepperd's method."""
    q = torch.zeros((R.size(0), 4), device=R.device)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cond1 = trace > 0
    
    if cond1.any():
        S = torch.sqrt(trace[cond1] + 1.0) * 2
        q[cond1, 0] = 0.25 * S
        q[cond1, 1] = (R[:, 2, 1][cond1] - R[:, 1, 2][cond1]) / S
        q[cond1, 2] = (R[:, 0, 2][cond1] - R[:, 2, 0][cond1]) / S
        q[cond1, 3] = (R[:, 1, 0][cond1] - R[:, 0, 1][cond1]) / S
    
    cond2 = ~cond1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if cond2.any():
        S = torch.sqrt(1.0 + R[:, 0, 0][cond2] - R[:, 1, 1][cond2] - R[:, 2, 2][cond2]) * 2
        q[cond2, 0] = (R[:, 2, 1][cond2] - R[:, 1, 2][cond2]) / S
        q[cond2, 1] = 0.25 * S
        q[cond2, 2] = (R[:, 0, 1][cond2] + R[:, 1, 0][cond2]) / S
        q[cond2, 3] = (R[:, 0, 2][cond2] + R[:, 2, 0][cond2]) / S
    
    cond3 = ~cond1 & ~cond2 & (R[:, 1, 1] > R[:, 2, 2])
    if cond3.any():
        S = torch.sqrt(1.0 + R[:, 1, 1][cond3] - R[:, 0, 0][cond3] - R[:, 2, 2][cond3]) * 2
        q[cond3, 0] = (R[:, 0, 2][cond3] - R[:, 2, 0][cond3]) / S
        q[cond3, 1] = (R[:, 0, 1][cond3] + R[:, 1, 0][cond3]) / S
        q[cond3, 2] = 0.25 * S
        q[cond3, 3] = (R[:, 1, 2][cond3] + R[:, 2, 1][cond3]) / S
    
    cond4 = ~cond1 & ~cond2 & ~cond3
    if cond4.any():
        S = torch.sqrt(1.0 + R[:, 2, 2][cond4] - R[:, 0, 0][cond4] - R[:, 1, 1][cond4]) * 2
        q[cond4, 0] = (R[:, 1, 0][cond4] - R[:, 0, 1][cond4]) / S
        q[cond4, 1] = (R[:, 0, 2][cond4] + R[:, 2, 0][cond4]) / S
        q[cond4, 2] = (R[:, 1, 2][cond4] + R[:, 2, 1][cond4]) / S
        q[cond4, 3] = 0.25 * S
    
    return q

def transform_gs_ply_full(input_ply, md_link_pose, rgb_scale=None):
    """Transform Gaussian Splatting model with optional RGB scaling."""
    gaussians = GaussianModel(0, 32)
    gaussians.load_ply(input_ply)
    if rgb_scale is not None:
        from ..utils.sh_utils import C0
        sh_dc, rgb = gaussians._features_dc, gaussians._features_dc * C0 + 0.5
        rgb_scale_tensor = torch.tensor(rgb_scale, dtype=torch.float32, device=rgb.device).view(1, 1, 3)
        rgb_adjusted = torch.clamp(rgb * rgb_scale_tensor, 0.0, 1.0)
        with torch.no_grad():
            gaussians._features_dc.copy_((rgb_adjusted - 0.5) / C0)
    mdh_tensor = torch.from_numpy(md_link_pose).float().to(gaussians._xyz.device)
    R, t = mdh_tensor[:3, :3], mdh_tensor[:3, 3]
    transformed_xyz = gaussians._xyz @ R.T + t
    rot_matrices = build_rotation_tensor(gaussians._rotation)
    transformed_rot_mat = torch.bmm(R.expand(rot_matrices.size(0), -1, -1), rot_matrices)
    transformed_quat = rotmat_to_quaternion_tensor(transformed_rot_mat)
    with torch.no_grad():
        gaussians._xyz.copy_(transformed_xyz)
        gaussians._rotation.copy_(transformed_quat)
    return gaussians



def transform_gs_ply_full_object(input_ply, objpose_matrix, rgb_scale=None):
    """Transform object Gaussian Splatting model with optional RGB scaling."""
    gaussians = GaussianModel(0)
    gaussians.load_ply(input_ply)
    if rgb_scale is not None:
        from ..utils.sh_utils import C0
        sh_dc, rgb = gaussians._features_dc, gaussians._features_dc * C0 + 0.5
        rgb_scale_tensor = torch.tensor(rgb_scale, dtype=torch.float32, device=rgb.device).view(1, 1, 3)
        rgb_adjusted = torch.clamp(rgb * rgb_scale_tensor, 0.0, 1.0)
        with torch.no_grad():
            gaussians._features_dc.copy_((rgb_adjusted - 0.5) / C0)
    objpose_tensor = torch.from_numpy(objpose_matrix).float().to(gaussians._xyz.device)
    R_obj2base, t_obj2base = objpose_tensor[:3, :3], objpose_tensor[:3, 3]
    transformed_xyz = gaussians._xyz @ R_obj2base.T + t_obj2base
    rot_matrices = build_rotation_tensor(gaussians._rotation)
    transformed_rot_mat = torch.bmm(R_obj2base.expand(rot_matrices.size(0), -1, -1), rot_matrices)
    transformed_quat = rotmat_to_quaternion_tensor(transformed_rot_mat)
    with torch.no_grad():
        gaussians._rotation.copy_(transformed_quat)
        gaussians._xyz.copy_(transformed_xyz)
    return gaussians



def merge_3dgs_models(gaussians1, gaussians2):
    """Merge two Gaussian Splatting models into one."""
    def check_tensor(name, t1, t2):
        if t1.shape[1:] != t2.shape[1:]:
            raise ValueError(f"Shape mismatch: {name} ({t1.shape[1:]} vs {t2.shape[1:]})")
    
    check_tensor("features_dc", gaussians1._features_dc, gaussians2._features_dc)
    check_tensor("features_rest", gaussians1._features_rest, gaussians2._features_rest)
    check_tensor("opacity", gaussians1._opacity, gaussians2._opacity)
    check_tensor("scaling", gaussians1._scaling, gaussians2._scaling)
    check_tensor("rotation", gaussians1._rotation, gaussians2._rotation)
    
    merged_gaussians = GaussianModel(0)
    merged_gaussians._xyz = torch.cat([gaussians1._xyz, gaussians2._xyz], dim=0)
    merged_gaussians._features_dc = torch.cat([gaussians1._features_dc, gaussians2._features_dc], dim=0)
    merged_gaussians._features_rest = torch.cat([gaussians1._features_rest, gaussians2._features_rest], dim=0)
    merged_gaussians._opacity = torch.cat([gaussians1._opacity, gaussians2._opacity], dim=0)
    merged_gaussians._scaling = torch.cat([gaussians1._scaling, gaussians2._scaling], dim=0)
    merged_gaussians._rotation = torch.cat([gaussians1._rotation, gaussians2._rotation], dim=0)
    merged_gaussians.max_radii2D = torch.cat([gaussians1.max_radii2D, gaussians2.max_radii2D], dim=0)
    return merged_gaussians

if __name__ == "__main__":
    # Example usage:
    gaussian_sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(gaussian_sim_dir, "assets", "robot", "robot_urdf", "panda", "panda_v2.urdf")
    joint_states = {
        'panda_joint1': 0.0,
        'panda_joint2': -0.0,
        'panda_joint3': 0.0,
        'panda_joint4': 0.0,
        'panda_joint5': 0.0,
        'panda_joint6': -0.0,
        'panda_joint7': 0.0,
        # Gripper joints (optional)
        'panda_finger_joint1': 0.00,
        'panda_finger_joint2': 0.00
    }
    
    T_root = np.eye(4)  # Identity transform for the root link

    # Instantiate the kinematics class
    fk = URDFKinematics(urdf_path, joint_states, T_root)

    angle = [-28.21023437,11.33585937,32.60898438,-153.01265625,123.04984375, 73.21445313,2.1946875]
    angle = np.array(angle, dtype=np.float32)/180.0*np.pi
    angle = np.concatenate((angle, [0.01, 0.01]), axis=0)
    joint_states = {
        'panda_joint1': angle[0],
        'panda_joint2': angle[1],
        'panda_joint3': angle[2],
        'panda_joint4': angle[3],
        'panda_joint5': angle[4],
        'panda_joint6': angle[5],
        'panda_joint7': angle[6],
        'panda_finger_joint1': angle[7],
        'panda_finger_joint2': angle[8]
    }

    link_trans_poses = fk.get_new_link_tran_poses(joint_states)
    
    ply_dir = os.path.join(gaussian_sim_dir, "assets", "robot", "robot_ply", "ply_dynamic_input")
    scene_ply_path = os.path.join(gaussian_sim_dir, "assets", "scenes", "scene_blue_sapien", "point_cloud.ply")
    
    merged_gaussians = GaussianModel(0)
    merged_gaussians.load_ply(scene_ply_path)

    for link_name, T_link in link_trans_poses.items():
        input_ply = os.path.join(ply_dir, f"{link_name}.ply")
        gaussians = transform_gs_ply_full(input_ply, T_link)
        merged_gaussians = merge_3dgs_models(merged_gaussians, gaussians)
        
    fk.visualize_full_robot(view_link_pose=True, view_robot=True)

    # fk.visualize_all_links(axis_length=0.2)
    

