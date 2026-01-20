import torch
import numpy as np
import torch.nn as nn
from ..utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    """
    Camera module for 3D Gaussian Splatting rendering.
    """
    
    def __init__(self, uid, R, T, FovX, FovY, width, height, trans=None, scale=1.0, data_device="cuda"):
        super(Camera, self).__init__()

        self.uid = uid
        self.R = R
        self.T = T
        self.FovX = FovX
        self.FovY = FovY
        self.width = width
        self.height = height

        # Initialize device
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")
        
        # Set transformation parameters
        if trans is None:
            trans = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.trans = trans
        self.scale = scale
        self.zfar = 100.0
        self.znear = 0.01
        
        # Compute transformation matrices
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FovX, fovY=self.FovY).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    """
    Minimal camera class with essential parameters only.
    """
    
    def __init__(self, width, height, FovY, FovX, znear, zfar, world_view_transform, full_proj_transform):
        self.width = width
        self.height = height
        self.FovY = FovY
        self.FovX = FovX
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = torch.inverse(self.world_view_transform)[3][:3]
