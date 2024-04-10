import numpy as np

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


def cal_poses_diff_fro(poses1_dic, poses2_dic):
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

            diff = np.linalg.norm(pose1 - pose2, 'fro')

            count_comparisons += 1

    if count_comparisons > 0:
        # Calculate median differences
        median_diff = np.median(diff)

        # Calculate unified difference
        unified_diff = median_diff / count_comparisons
        print(f"Median difference: {unified_diff:.4f}")
    else:
        unified_diff = 0.0

    print(f"Total number of skipped poses: {count_skip}")
    print(f"Total comparisons: {count_comparisons}")
    return unified_diff
    