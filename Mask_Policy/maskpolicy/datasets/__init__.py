# Direct imports - these don't cause circular dependencies
from maskpolicy.datasets.robot_dataset import ConfigurableRobotDataset, collate_fn
from maskpolicy.datasets.constants import get_tensor_keys_from_data, CAM_2_BASE

# For backward compatibility
RobotDataset = ConfigurableRobotDataset

# make_dataset is imported directly from factory
# factory.py uses lazy import for TrainPipelineConfig to avoid circular deps
from maskpolicy.datasets.factory import make_dataset

__all__ = [
    'ConfigurableRobotDataset',
    'RobotDataset',  # Alias for backward compatibility
    'collate_fn', 
    'make_dataset',
    'get_tensor_keys_from_data',
    'CAM_2_BASE',
]

