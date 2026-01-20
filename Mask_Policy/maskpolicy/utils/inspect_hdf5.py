#!/usr/bin/env python
"""
Utility script to inspect HDF5 file structure.

This script displays the hierarchical structure of an HDF5 file, including:
- Groups and datasets
- Dataset shapes, dtypes, and sizes
- Attributes
- Data statistics (min, max, mean for numeric data)
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_dataset_size(dataset: h5py.Dataset) -> int:
    """Get the size of a dataset in bytes."""
    return dataset.nbytes


def get_data_statistics(data: np.ndarray) -> dict[str, Any]:
    """Get statistics for numeric data."""
    stats = {}
    if data.size == 0:
        return stats
    
    if np.issubdtype(data.dtype, np.number):
        stats['min'] = float(np.min(data))
        stats['max'] = float(np.max(data))
        stats['mean'] = float(np.mean(data))
        if data.size > 1:
            stats['std'] = float(np.std(data))
        if data.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            stats['unique_values'] = int(len(np.unique(data)))
    else:
        stats['type'] = 'non-numeric'
    
    return stats


def print_attributes(obj: h5py.Group | h5py.Dataset, indent: str = "  ") -> None:
    """Print attributes of an HDF5 object."""
    if len(obj.attrs) > 0:
        print(f"{indent}Attributes:")
        for key in obj.attrs.keys():
            value = obj.attrs[key]
            if isinstance(value, (bytes, np.bytes_)):
                try:
                    value = value.decode('utf-8')
                except:
                    value = str(value)
            elif isinstance(value, np.ndarray):
                if value.size <= 10:
                    value = value.tolist()
                else:
                    value = f"array(shape={value.shape}, dtype={value.dtype})"
            print(f"{indent}  {key}: {value}")


def inspect_group(group: h5py.Group, name: str = "", max_depth: int = 10, current_depth: int = 0, 
                  show_stats: bool = False, show_data: bool = False, max_data_samples: int = 5) -> None:
    """
    Recursively inspect an HDF5 group.
    
    Args:
        group: HDF5 group to inspect
        name: Name of the group
        max_depth: Maximum depth to traverse
        current_depth: Current depth in the hierarchy
        show_stats: Whether to show data statistics
        show_data: Whether to show sample data
        max_data_samples: Maximum number of data samples to show
    """
    indent = "  " * current_depth
    
    if current_depth == 0:
        print(f"\n{'='*60}")
        print(f"HDF5 File Structure: {name}")
        print(f"{'='*60}\n")
    else:
        print(f"{indent}Group: {name}")
        print_attributes(group, indent)
    
    if current_depth >= max_depth:
        print(f"{indent}  ... (max depth reached)")
        return
    
    # List all items in the group
    items = list(group.keys())
    
    for item_name in sorted(items):
        item = group[item_name]
        item_indent = "  " * (current_depth + 1)
        
        if isinstance(item, h5py.Group):
            print(f"{item_indent}📁 {item_name}/")
            inspect_group(item, item_name, max_depth, current_depth + 1, show_stats, show_data, max_data_samples)
        elif isinstance(item, h5py.Dataset):
            print(f"{item_indent}📄 {item_name}")
            print(f"{item_indent}  Shape: {item.shape}")
            print(f"{item_indent}  Dtype: {item.dtype}")
            size = get_dataset_size(item)
            print(f"{item_indent}  Size: {format_size(size)}")
            
            # Print attributes
            print_attributes(item, item_indent)
            
            # Show statistics if requested
            if show_stats and item.size > 0:
                try:
                    data = item[:]
                    stats = get_data_statistics(data)
                    if stats:
                        print(f"{item_indent}  Statistics:")
                        for key, value in stats.items():
                            if isinstance(value, float):
                                print(f"{item_indent}    {key}: {value:.6f}")
                            else:
                                print(f"{item_indent}    {key}: {value}")
                except Exception as e:
                    print(f"{item_indent}  ⚠️  Could not load data for statistics: {e}")
            
            # Show sample data if requested
            if show_data and item.size > 0:
                try:
                    data = item[:]
                    if data.size <= max_data_samples:
                        print(f"{item_indent}  Data: {data}")
                    else:
                        # Show first few elements
                        flat_data = data.flatten()
                        sample = flat_data[:max_data_samples]
                        print(f"{item_indent}  Data sample (first {max_data_samples}): {sample}")
                        print(f"{item_indent}  ... (total {data.size} elements)")
                except Exception as e:
                    print(f"{item_indent}  ⚠️  Could not load data: {e}")


def inspect_hdf5(file_path: str, data_group: str = None, max_depth: int = 10, 
                 show_stats: bool = False, show_data: bool = False, max_data_samples: int = 5) -> None:
    """
    Inspect an HDF5 file structure.
    
    Args:
        file_path: Path to the HDF5 file
        data_group: Optional group name to inspect (default: inspect entire file)
        max_depth: Maximum depth to traverse
        show_stats: Whether to show data statistics
        show_data: Whether to show sample data
        max_data_samples: Maximum number of data samples to show
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n📁 HDF5 File: {file_path}")
            print(f"   File size: {format_size(file_path.stat().st_size)}")
            
            if data_group:
                if data_group in f:
                    group = f[data_group]
                    inspect_group(group, data_group, max_depth, 0, show_stats, show_data, max_data_samples)
                else:
                    print(f"❌ Error: Group '{data_group}' not found in file")
                    print(f"   Available top-level groups: {list(f.keys())}")
                    sys.exit(1)
            else:
                inspect_group(f, str(file_path), max_depth, 0, show_stats, show_data, max_data_samples)
            
            print(f"\n{'='*60}")
            print("✅ Inspection complete")
            print(f"{'='*60}\n")
    
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 file structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection
  python -m maskpolicy.utils.inspect_hdf5 data.hdf5
  
  # Inspect specific group
  python -m maskpolicy.utils.inspect_hdf5 data.hdf5 --group data
  
  # Show statistics
  python -m maskpolicy.utils.inspect_hdf5 data.hdf5 --stats
  
  # Show sample data
  python -m maskpolicy.utils.inspect_hdf5 data.hdf5 --data
  
  # Limit depth
  python -m maskpolicy.utils.inspect_hdf5 data.hdf5 --max-depth 3
        """
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to HDF5 file"
    )
    parser.add_argument(
        "--group", "-g",
        type=str,
        default=None,
        help="Specific group to inspect (default: inspect entire file)"
    )
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=10,
        help="Maximum depth to traverse (default: 10)"
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show data statistics (min, max, mean, std)"
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Show sample data from datasets"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum number of data samples to show (default: 5)"
    )
    
    args = parser.parse_args()
    
    inspect_hdf5(
        args.file,
        data_group=args.group,
        max_depth=args.max_depth,
        show_stats=args.stats,
        show_data=args.data,
        max_data_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

