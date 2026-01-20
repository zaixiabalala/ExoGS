import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import ruptures as rpt
from typing import List, Union
logger = logging.getLogger(__name__)


def detect_grasp_frame(gripper_width: Union[np.ndarray, List[float]]) -> dict:
    """
    detect grasp frame from gripper width sequence
    
    Args:
        gripper_width: gripper width sequence
        min_grasp_change_threshold: minimum grasp change threshold, default is 0.003 (3mm)

    Returns:
        Dict[str, List[int]]: 
            'grasp_frame_idx_list': List[int],  # grasp frame index list
            'release_frame_idx_list': List[int]    # release frame index list
    """
    if isinstance(gripper_width, np.ndarray):
        signal = gripper_width.copy()
    elif isinstance(gripper_width, list) and all(isinstance(item, float) for item in gripper_width):
        signal = np.array(gripper_width).copy()
    else:
        e = f"Invalid gripper width type: {type(gripper_width)}";logger.error(e);raise ValueError(e) from None
    
    data_range = np.max(signal) - np.min(signal)
    data_std = np.std(signal)
    
    # adaptive penalty based on data variance and range
    base_pen = data_std * 10
    pen = max(0.1, min(base_pen, 2.0))

    logger.debug(f"Gripper width statistics: range=[{np.min(signal):.4f}, {np.max(signal):.4f}], "
                 f"std={data_std:.4f}, using penalty={pen:.2f}")
    
    # use PELT algorithm to detect change points, model="l2": using L2 norm, min_size=10: minimum segment length is 10
    algo = rpt.Pelt(model="l2", min_size=10).fit(signal)
    # try different penalty values, if no change points are detected, gradually decrease the penalty
    penalty_candidates = [pen, pen * 0.5, pen * 0.1, 0.1, 0.05, 0.01] if pen > 0.1 else [pen, pen * 0.5, 0.01]
    change_points = None
    
    for test_pen in penalty_candidates:
        change_points = algo.predict(pen=test_pen)
        if len(change_points) > 1:  # at least 2 change points
            break
    
    if change_points is None or len(change_points) <= 1:
        e = f"Failed to detect change points! data range={data_range:.4f}, std={data_std:.4f}";logger.error(e);raise ValueError(e) from None
    
    # exclude the last change point, (the end of the sequence)
    change_points = change_points[:-1]
    
    if len(change_points) == 0:
        e = f"No remaining change points after excluding the last change point";logger.error(e);raise ValueError(e) from None
    
    logger.debug(f"Detected {len(change_points)} change points: {change_points}")

    # determine the change direction
    close_points = []  # open to close
    open_points = []   # close to open
    for cp in change_points:
        # check the mean of the change before and after
        before_mean = np.mean(signal[max(0, cp-20):cp])
        after_mean = np.mean(signal[cp:min(len(signal), cp+20)])
        
        change_magnitude = abs(before_mean - after_mean)

        if before_mean > after_mean:
            close_points.append(cp)
        else:
            open_points.append(cp)
    
    logger.debug(f"Close points: {close_points}, Open points: {open_points}")
    
    # if len(close_points) == 0 or len(open_points) == 0:
        # e = f"No grasp or release frame detected! close points={len(close_points)}, open points={len(open_points)}";logger.error(e);raise ValueError(e) from None
    
    # # 临时处理！！！
    # open_points = [len(signal) - 15]
    # if len(close_points) == 0:
    #     e = f"No close point detected! close points={len(close_points)}";logger.error(e);raise ValueError(e) from None

    # filter out short errors (30 frames or less of close and open)
    min_duration = 20  # minimum duration (frames)
    filtered_close_points = []
    filtered_open_points = []
    
    # for each close point, find the nearest open point after it, check the duration
    used_open_indices = set()  # record the indices of the used open points
    for close_point in close_points:
        best_open_point = None
        best_open_idx = None
        best_duration = float('inf')
        # find the nearest open point after the close point
        for idx, open_point in enumerate(open_points):
            if open_point > close_point and idx not in used_open_indices:
                duration = open_point - close_point
                if duration < best_duration:
                    best_duration = duration
                    best_open_point = open_point
                    best_open_idx = idx
        # if a matching open point is found
        if best_open_point is not None:
            if best_duration >= min_duration:
                filtered_close_points.append(close_point)
                # for release frame, find the position where the width starts to increase
                # in the [close_point, best_open_point] interval, find the position where the width is smallest
                segment_start = max(0, close_point)
                segment_end = min(len(signal), best_open_point + 1)
                segment_signal = signal[segment_start:segment_end]
                
                # find the position where the width is smallest (relative to the entire sequence)
                min_width_idx_in_segment = np.argmin(segment_signal)
                min_width_idx = segment_start + min_width_idx_in_segment
                min_width_value = segment_signal[min_width_idx_in_segment]
                
                # from the minimum width position, search forward (towards best_open_point) to find the position where the width starts to increase
                # use the difference to detect the trend of width starting to increase
                # calculate the difference from the minimum position to best_open_point
                search_start = min_width_idx
                search_end = segment_end
                
                # use the sliding window to detect the trend of width starting to increase
                window_size = 8  # 8 frames window
                actual_release_point = search_start
                
                for i in range(search_start, search_end - window_size + 1):
                    window = signal[i:i+window_size]
                    # check if the width is continuously increasing (all differences are positive)
                    diffs = np.diff(window)
                    if np.all(diffs > 0) and len(diffs) > 0:
                        # check if the increase is large enough (to avoid noise)
                        total_increase = window[-1] - window[0]
                        if total_increase > 0.005:  # at least increase 0.005 (5mm)
                            actual_release_point = i
                            break
                
                # if no position where the width starts to increase is found, use the threshold method as a backup
                if actual_release_point == search_start:
                    release_threshold_relative = min_width_value * 0.05
                    release_threshold_absolute = 0.01
                    release_threshold = min(release_threshold_relative, release_threshold_absolute)
                    
                    for i in range(search_start, search_end):
                        if signal[i] - min_width_value >= release_threshold:
                            actual_release_point = i
                            break
                
                # ensure the release frame is after the grasp frame
                if actual_release_point <= close_point:
                    actual_release_point = close_point + 1
                
                filtered_open_points.append(actual_release_point)
                used_open_indices.add(best_open_idx)
                
                logger.debug(f"Release frame adjustment: original={best_open_point}, min_width_idx={min_width_idx}, "
                           f"min_width={min_width_value:.4f}, adjusted_release={actual_release_point}, "
                           f"width_at_release={signal[actual_release_point]:.4f}")
            else:
                logger.warning(f"Filtered out short error: close point={close_point}, open point={best_open_point}, duration={best_duration} frames")
        else:
            # no matching open point found
            logger.warning(f"No matching open point found for close point={close_point}, skipping")
    
    if len(filtered_close_points) == 0 or len(filtered_open_points) == 0:
        e = f"No valid grasp or release frame detected after filtering out short errors! original close points={len(close_points)}, original open points={len(open_points)}, filtered close points={len(filtered_close_points)}, filtered open points={len(filtered_open_points)}";logger.error(e);raise ValueError(e) from None
    
    logger.debug(f"Grasp frame index list: {filtered_close_points}, Release frame index list: {filtered_open_points}")

    return {'grasp_frame_idx_list': filtered_close_points, 'release_frame_idx_list': filtered_open_points}


def get_fixed_relative_pose(
    obj_pose_grasp: np.ndarray,
    ee_pose_grasp: np.ndarray,
) -> np.ndarray:
    """
    get the fixed relative pose of the object to the eef
    
    Args:
        obj_pose_grasp: object pose in grasp frame (4x4 matrix)
        ee_pose_grasp: eef pose in grasp frame (4x4 matrix)

    Returns:
        fixed relative pose (4x4 matrix), represents the pose of the object relative to the eef
    """
    if not obj_pose_grasp.shape == (4, 4):
        e = f"Object pose shape error: expected 4x4, actual {obj_pose_grasp.shape}";logger.error(e);raise ValueError(e) from None
    if not ee_pose_grasp.shape == (4, 4):
        e = f"Eef pose shape error: expected 4x4, actual {ee_pose_grasp.shape}";logger.error(e);raise ValueError(e) from None
    
    # copy the object pose to avoid modifying the original data
    obj_pose_adjusted = obj_pose_grasp.copy()
    
    # project the object position to the plane perpendicular to the eef x-axis, through the eef center
    ee_center = ee_pose_grasp[:3, 3]
    x_axis = ee_pose_grasp[:3, 0]
    obj_pos = obj_pose_grasp[:3, 3]
    vec_to_center = obj_pos - ee_center
    x_component = np.dot(vec_to_center, x_axis) * x_axis
    projected_pos = obj_pos - x_component
    obj_pose_adjusted[:3, 3] = projected_pos

    # calculate the fixed relative pose of the object to the eef
    ee_pose_inv_grasp = np.linalg.inv(ee_pose_grasp)
    fixed_relative_pose = ee_pose_inv_grasp @ obj_pose_adjusted

    return fixed_relative_pose

def fix_object_pose_to_eef(
    object_pose_list: List[np.ndarray],
    eef_pose_list: List[np.ndarray],
    reference_frame_idx: int,
    fix_start_frame_idx: int,
    fix_end_frame_idx: int,
) -> List[np.ndarray]:
    """
    fix the object pose to the eef pose

    Args:
        object_pose_list: object pose list (List of 4x4 matrices)
        eef_pose_list: eef pose list (List of 4x4 matrices)
        reference_frame_idx: frame index to calculate the fixed relative pose
        fix_start_frame_idx: frame index to start fixing
        fix_end_frame_idx: frame index to end fixing

    Returns:
        object pose list (List of 4x4 matrices)
    """
    if len(object_pose_list) != len(eef_pose_list):
        e = f"Data length mismatch: object_pose_list={len(object_pose_list)}, eef_pose_list={len(eef_pose_list)}";logger.error(e);raise ValueError(e) from None
    
    obj_pose_grasp = np.array(object_pose_list[reference_frame_idx]).copy()
    ee_pose_grasp = np.array(eef_pose_list[reference_frame_idx]).copy()
    fixed_relative_pose = get_fixed_relative_pose(obj_pose_grasp=obj_pose_grasp, ee_pose_grasp=ee_pose_grasp)
    
    # overwrite the object pose from fix_start to fix_end
    for i in range(fix_start_frame_idx, fix_end_frame_idx + 1):
        ee_pose = eef_pose_list[i]
        new_obj_pose = ee_pose @ fixed_relative_pose
        object_pose_list[i] = new_obj_pose
    
    logger.debug(f"Fixed object pose to eef pose: reference frame {reference_frame_idx}, fix range [{fix_start_frame_idx}, {fix_end_frame_idx}], total {fix_end_frame_idx - fix_start_frame_idx + 1} frames")
    
    return object_pose_list
