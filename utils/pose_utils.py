import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def is_similar(pose1, pose2, threshold_trans=0.05, threshold_rot=0.15):
    """
    Determine if two poses are similar based on translational and rotational thresholds.

    Args:
        pose1 (np.array): The first pose as a 4x4 transformation matrix.
        pose2 (np.array): The second pose as a 4x4 transformation matrix.
        threshold_trans (float): The translational threshold in meters.
        threshold_rot (float): The rotational threshold in radians.

    Returns:
        flag: True if the poses are similar, False otherwise.
    """

    pose1 = np.array(pose1)
    pose2 = np.array(pose2)

    # Extract translational components and calculate difference
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]
    translational_diff = np.linalg.norm(t1 - t2)

    # Extract rotational components and calculate geodesic distance
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    trace = np.trace(np.dot(np.transpose(R1), R2))
    theta = np.arccos(max(min((trace - 1) / 2, 1), -1))

    # Determine if the poses are similar
    if translational_diff <= threshold_trans and theta <= threshold_rot:
        return True
    else:
        return False

def cal_poses_diff(poses1_dic, poses2_dic):
    """
    Calculates a unified, normalized difference between two dictionaries of poses,
    taking into account both translational and rotational components, with robustness improvements
    and normalization across the number of comparisons.

    Args:
        poses1 (dict): The first dictionary of poses.
        poses2 (dict): The second dictionary of poses.

    Returns:
        float: The unified, normalized difference between the poses, normalized by the count of comparisons.

    Raises:
        AssertionError: If the input dictionaries do not have the same keys or if the lists for a key do not have the same length.
    """
    assert set(poses1_dic.keys()) == set(poses2_dic.keys()), "Input dictionaries must have the same keys"

    translational_diffs = []
    rotational_diffs = []
    count_skip = 0
    count_comparisons = 0

    for key in poses1_dic.keys():
        list1 = np.array(poses1_dic[key])
        list2 = np.array(poses2_dic[key])
        assert len(list1) == len(list2), f"Lists for key '{key}' must have the same length"

        for pose1, pose2 in zip(list1, list2):
            if np.array_equal(pose1, np.eye(4)) or np.array_equal(pose2, np.eye(4)):
                count_skip += 1
                continue

            # Extract translational components and calculate difference
            t1 = pose1[:3, 3]
            t2 = pose2[:3, 3]
            translational_diffs.append(np.linalg.norm(t1 - t2))

            # Extract rotational components and calculate geodesic distance
            R1 = pose1[:3, :3]
            R2 = pose2[:3, :3]
            trace = np.trace(np.dot(np.transpose(R1), R2))
            theta = np.arccos(max(min((trace - 1) / 2, 1), -1))
            rotational_diffs.append(theta)

            count_comparisons += 1

    if count_comparisons > 0:
        # Calculate median differences
        median_translational_diff = np.median(translational_diffs)
        median_rotational_diff = np.median(rotational_diffs)

        # Calculate unified difference
        unified_diff = (median_translational_diff + median_rotational_diff) / count_comparisons
        print(f"Median translational difference: {median_translational_diff:.4f}")
        print(f"Median rotational difference: {median_rotational_diff:.4f}")
    else:
        unified_diff = 0.0

    print(f"Total number of skipped poses: {count_skip}")
    print(f"Total comparisons: {count_comparisons}")
    return unified_diff

def interpolate_transform(start_T, end_T, steps):
    """
    Performs linear interpolation on transformation matrices.

    Args:
        start_T (np.array): The starting 4x4 transformation matrix.
        end_T (np.array): The ending 4x4 transformation matrix.
        steps (int): The number of intermediate matrices to be outputted.

    Returns:
        list: A list of interpolated 4x4 transformation matrices, including the start and end matrices.
    """
    start_T = np.array(start_T)
    end_T = np.array(end_T)

    # Initialize the list of interpolated matrices, starting with the start matrix
    interpolated_matrices = [start_T]
    
    # Calculate the step increment for translation and rotation (quaternion)
    start_translation, end_translation = start_T[:3, 3], end_T[:3, 3]
    translation_step = (end_translation - start_translation) / (steps + 1)
    
    start_rotation = R.from_matrix(start_T[:3, :3])
    end_rotation = R.from_matrix(end_T[:3, :3])

    # Create a spherical linear interpolation object
    key_rots = R.from_quat(np.vstack([start_rotation.as_quat(), end_rotation.as_quat()]))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)

    times = np.linspace(0, 1, steps + 2)
    interp_rots = slerp(times)
    
    for step in range(1, steps + 1):
        # Interpolate translation
        interpolated_translation = start_translation + translation_step * step
        
        # Construct the interpolated transformation matrix
        interpolated_T = np.eye(4)
        interpolated_T[:3, :3] = interp_rots[step].as_matrix()
        interpolated_T[:3, 3] = interpolated_translation
        
        # Append the interpolated transformation matrix to the list
        interpolated_matrices.append(interpolated_T)
    
    # Append the end matrix to the list of interpolated matrices
    interpolated_matrices.append(end_T)

    interpolated_matrices.pop(0) # remove the first matrix
    interpolated_matrices.pop(-1) # remove the last matrix
     
    return interpolated_matrices
    
def ave_rotation(R_list):
    """
    Computes the average rotation matrix from a list of rotation matrices.
    
    Args:
        rotations (list of np.array): A list of 3x3 rotation matrices.
    
    Returns:
        np.array: The average rotation matrix.
    """
    # Step a: Compute the 'mean matrix' M of the rotations
    M = sum(R_list) / len(R_list)
    
    # Step b: Take the SVD of that mean matrix M
    U, _, Vt = np.linalg.svd(M)
    
    # Step c: Compute the rotation closest to M
    Q = np.dot(U, Vt)
    
    # Ensure Q is a proper rotation matrix by checking its determinant
    # and correcting if necessary
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1  # Flip last column of U if determinant of Q is negative
        Q = np.dot(U, Vt)
    
    return Q

def ave_pose(T_list):
    """
    Calculate the average pose of a list of poses.

    Args:
        T_list (list of np.array): A list of 4x4 transformation matrices.

    Returns:
        np.array: The average pose as a 4x4 transformation matrix.
    """

    # Extract translational components and calculate average
    ave_t = np.mean([np.array(T)[:3, 3] for T in T_list], axis=0)

    # Extract rotational components and calculate average
    ave_R = ave_rotation([np.array(T)[:3, :3] for T in T_list])

    ave_T = np.eye(4)
    ave_T[:3, :3] = ave_R
    ave_T[:3, 3] = ave_t

    return ave_T