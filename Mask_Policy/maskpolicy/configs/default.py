from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING

# ImageTransformsConfig is imported at runtime when needed
if TYPE_CHECKING:
    from maskpolicy.datasets.transforms import ImageTransformsConfig
else:
    ImageTransformsConfig = Any  # Placeholder for runtime


@dataclass
class HDF5FieldConfig:
    """Configuration for a single HDF5 field."""
    hdf5_key: str  # Key in HDF5 file (e.g., 'color_imgs', 'robot_xyzs')
    shape: Tuple[int, ...]  # Expected shape per sample
    dtype: str = "float32"  # Data type
    augmentation: Optional[str] = None  # "pose", "color", "noise", None
    required: bool = True
    dims: int = 0  # 1 for 1D, 2 for 2D, 3 for 3D
    transform: Optional[str] = None  # "quaternion_to_6d", "xyz_to_delta", None


@dataclass
class DatasetFieldConfig:
    """Configuration for a dataset output field."""
    output_key: str  # Key in output dict (e.g., 'observation.images.cam_0')
    hdf5_fields: List[str]  # List of HDF5FieldConfig keys
    shape: Tuple[int, ...]  # Output shape
    dtype: str = "float32"
    # Default value for output fields not in delta_indices (optional)
    default_value: Optional[Union[float, int]] = None
    # Normalization mode for this field: "imagenet", "meanstd", "minmax", None
    # Used to automatically infer normalization_mapping in policy
    normalization: Optional[str] = None
    normalization_params: Optional[Dict[str, Any]] = None  # Precomputed normalization params


@dataclass
class HDF5DatasetConfig:
    """HDF5 dataset structure configuration."""
    hdf5_data_group: str = "data"
    hdf5_fields: Dict[str, HDF5FieldConfig] = field(default_factory=dict)
    # Input fields: what the model receives (observations)
    input_fields: Dict[str, DatasetFieldConfig] = field(default_factory=dict)
    # Output fields: what the model predicts (actions, masks, etc.)
    output_fields: Dict[str, DatasetFieldConfig] = field(default_factory=dict)
    output_format: str = "policy"  # "raw" or "policy"
    # Observation and action horizon (from policy config)
    n_obs_steps: int = 1  # Number of observation steps to use
    chunk_size: int = 100  # Number of action steps to predict
    # Delta indices for multi-step extraction (like lerobot)
    # Key: output field name (e.g., "action"), Value: list of delta indices (e.g., [0, 1, 2, ..., 49])
    delta_indices: Optional[Dict[str, List[int]]] = None
    # Whether to allow padding (Lerobot-style: all frames can be starting points, padding at boundaries)
    # If False (default): skip episodes that are too short, no padding needed
    # If True: allow all frames, compute padding masks at runtime (like lerobot)
    allow_padding: bool = False

    def validate(self):
        """Validate configuration."""
        # Check that all input_fields reference valid hdf5_fields
        for input_key, input_config in self.input_fields.items():
            for hdf5_field_key in input_config.hdf5_fields:
                if hdf5_field_key not in self.hdf5_fields:
                    raise ValueError(
                        f"Input field '{input_key}' references unknown HDF5 field '{hdf5_field_key}'. "
                        f"Available fields: {list(self.hdf5_fields.keys())}"
                    )

        # Check that all output_fields reference valid hdf5_fields
        for output_key, output_config in self.output_fields.items():
            for hdf5_field_key in output_config.hdf5_fields:
                if hdf5_field_key not in self.hdf5_fields:
                    raise ValueError(
                        f"Output field '{output_key}' references unknown HDF5 field '{hdf5_field_key}'. "
                    f"Available fields: {list(self.hdf5_fields.keys())}"
                )


@dataclass
class CustomDatasetConfig:
    """Custom HDF5 dataset configuration."""
    path: str
    # Flexible config-based approach - hdf5_config is required
    hdf5_config: Optional[HDF5DatasetConfig] = None


@dataclass
class DatasetConfig:
    """Dataset configuration for custom HDF5 datasets."""
    # Image transforms configuration (lerobot-style)
    image_transforms: Union["ImageTransformsConfig", Any] = field(
        default_factory=lambda: __import__('maskpolicy.datasets.transforms', fromlist=['ImageTransformsConfig']).ImageTransformsConfig())
    # Custom HDF5 dataset configuration (required)
    custom: CustomDatasetConfig | None = None


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    n_episodes: int = 50
    batch_size: int = 50

    def __post_init__(self):
        if self.batch_size > self.n_episodes:
            raise ValueError(
                f"eval.batch_size ({self.batch_size}) > eval.n_episodes ({self.n_episodes})"
            )
