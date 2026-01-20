import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import List


def read_angles_from_folder(folder_path: str) -> List[np.ndarray]:
    """
    从文件夹中读取所有npy文件中的角度数据，按文件名自动排序
    
    参数:
        folder_path: 包含npy文件的文件夹路径
    
    返回:
        角度数据列表，每个元素是一个numpy数组（通常是8位的关节角度）
    """
    assert os.path.exists(folder_path), f"[Error] 文件夹不存在: {folder_path}"
    assert os.path.isdir(folder_path), f"[Error] 路径不是文件夹: {folder_path}"

    angles_list: List[np.ndarray] = []
    
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    npy_files.sort()  # 按文件名排序
    
    if not npy_files:
        raise ValueError(f"[Error] 文件夹中没有找到npy文件: {folder_path}")
    
    for npy_file in npy_files:
        file_path = os.path.join(folder_path, npy_file)
        try:
            angles = np.load(file_path)
            angles_list.append(angles)                
        except Exception as e:
            raise ValueError(f"[Error] 读取文件失败 {npy_file}: {e}")
    
    assert len(angles_list) > 0, f"[Error] 文件夹中没有找到角度数据: {folder_path}"
    assert angles_list[0].shape == (8,), f"[Error] 每个数据的形状不正确: {angles_list[0].shape}"

    return angles_list


def write_angles_to_folder(angles_list: List[np.ndarray], 
                           folder_path: str, 
                           file_name_format: str = "angle_cam0_{i:05d}.npy"):
    """
    将角度数据列表写入文件夹
    
    参数:
        angles_list: 角度数据列表
        folder_path: 目标文件夹路径
        prefix: 文件名前缀，默认"angle_cam0_"
    """

    os.makedirs(folder_path, exist_ok=True)

    assert len(angles_list) > 0, f"[Error] 角度数据列表为空: {angles_list}"

    saved_count = 0
    for i, angles in enumerate(angles_list):
        file_name = file_name_format.format(i=i)
        file_path = os.path.join(folder_path, file_name)
        try:
            np.save(file_path, angles)
            saved_count += 1
        except Exception as e:
            raise ValueError(f"[Error] 写入失败 index={i}, path={file_path}: {e}")
    
    assert saved_count == len(angles_list), f"[Error] 写入的文件数量不正确: 期望 {len(angles_list)}, 实际 {saved_count}"


def shift_angles_from_frame(
    angles_list: List[np.ndarray],
    start_frame: int,
    shift_frames: int
) -> List[np.ndarray]:
    """
    从指定帧开始，将后续所有帧的关节角度前移或后移指定帧数
    
    参数:
        angles_list: 角度数据列表
        start_frame: 开始平移的帧索引（从0开始）
        shift_frames: 平移帧数
                     > 0: 后移，空出的前面帧用start_frame的数据填充
                     < 0: 前移，空出的后面帧用最后一帧的数据填充
                     = 0: 不变
    
    返回:
        处理后的角度数据列表
    """
    assert len(angles_list) > 0, f"[Error] 角度数据列表为空: {angles_list}"
    
    if start_frame < 0 or start_frame >= len(angles_list):
        raise ValueError(f"[Error] start_frame {start_frame} 超出范围 [0, {len(angles_list)-1}]")
    
    if shift_frames == 0:
        raise ValueError(f"[Error] shift_frames为0，不进行平移")
    
    # 创建结果列表，先复制原始数据
    result = [angles.copy() for angles in angles_list]
    total_frames = len(angles_list)
    
    if shift_frames > 0:
        # 从后往前遍历，将数据后移
        for i in range(total_frames - 1, start_frame - 1, -1):
            new_idx = i + shift_frames
            if new_idx < total_frames:
                # 如果新位置在范围内，移动数据
                result[new_idx] = angles_list[i].copy()
        # 用start_frame的数据填充前面空出的位置
        fill_data = angles_list[start_frame].copy()
        for i in range(start_frame, min(start_frame + shift_frames, total_frames)):
            result[i] = fill_data.copy()
        
    else:  # shift_frames < 0
        shift_amount = abs(shift_frames)
        # 从前往后遍历，将数据前移
        source_start = start_frame + shift_amount
        for i in range(start_frame, total_frames):
            source_idx = i + shift_amount
            if source_idx < total_frames:
                # 如果源位置在范围内，移动数据
                result[i] = angles_list[source_idx].copy()
            else:
                # 如果源位置超出范围，用最后一帧填充
                result[i] = angles_list[-1].copy()
        
        # 计算有多少帧被最后一帧填充
        filled_frames = max(0, shift_amount - (total_frames - start_frame))
        assert filled_frames == 0, f"[Error] 后面 {filled_frames} 帧用最后一帧数据填充"
    
    return result