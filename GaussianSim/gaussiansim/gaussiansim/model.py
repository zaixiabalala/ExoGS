import os
import json
import torch
import numpy as np
from torch import nn
from plyfile import PlyData

from ..utils.general_utils import strip_symmetric, build_scaling_rotation, inverse_sigmoid


class GaussianModel(nn.Module):
    """
    Gaussian Splatting model for 3D scene representation.
    """
    
    def setup_functions(self):
        """Setup activation functions for Gaussian parameters."""
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            return strip_symmetric(L @ L.transpose(1, 2))
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree):
        super(GaussianModel, self).__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()
        self.max_radii2D = torch.empty(0)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def load_ply(self, path, use_train_test_exp=False):
        """Load Gaussian Splatting model from PLY file."""
        plydata = PlyData.read(path)
        
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {
                    image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda()
                    for image_name in exposures
                }
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None
        
        # Extract position data
        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ), axis=1)
        
        # Extract opacity
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        # Extract DC features (spherical harmonics degree 0)
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        # Extract rest features (higher order spherical harmonics)
        extra_f_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split('_')[-1])
        )
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        # Extract scaling parameters
        scale_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split('_')[-1])
        )
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # Extract rotation parameters (quaternions)
        rot_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")],
            key=lambda x: int(x.split('_')[-1])
        )
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # Convert to PyTorch parameters
        self._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float).cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float).cuda().transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = torch.nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float).cuda().transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float).cuda().requires_grad_(True))
        self._scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float).cuda().requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float).cuda().requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
