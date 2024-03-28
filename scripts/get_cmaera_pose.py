'''
get the camera pose from the QR code transformation matrices
'''

import numpy as np
from scipy.spatial.transform import Rotation as R

def invert_transformation_matrix(T):
    """Inverts a transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def average_transformation_matrices(transformation_matrix_dic_list):
    """Averages a list of transformation matrices."""
    quaternions = []
    translations = []

    for transformation_matrix_dic in transformation_matrix_dic_list:
        if 'Label 1' in transformation_matrix_dic:
            T = transformation_matrix_dic['Label 1']
            R_mat = T[:3, :3]
            t_vec = T[:3, 3]
            quaternions.append(R.from_matrix(R_mat).as_quat())
            translations.append(t_vec)
    
    if not quaternions:
        print("No valid transformation matrices found.")
        return None

    # Average quaternions and translations
    mean_quaternion = np.mean(quaternions, axis=0)
    mean_translation = np.mean(translations, axis=0)

    # Convert the average quaternion back to a rotation matrix
    mean_rotation_matrix = R.from_quat(mean_quaternion).as_matrix()

    # Construct the average transformation matrix
    T_avg = np.eye(4)
    T_avg[:3, :3] = mean_rotation_matrix
    T_avg[:3, 3] = mean_translation

    return T_avg

def calculate_camera_pose_from_qr_code_avg(transformation_matrix_dic_list):
    """Calculates the camera pose from averaging QR code transformation matrices."""
    T_avg_qr_to_cam = average_transformation_matrices(transformation_matrix_dic_list)
    if T_avg_qr_to_cam is not None:
        T_avg_cam_to_qr = invert_transformation_matrix(T_avg_qr_to_cam)
        return T_avg_cam_to_qr
    else:
        return None

# Example usage:
# Assuming transformation_matrix_dic_list_camera1 and transformation_matrix_dic_list_camera2 are defined
# transformation_matrix_dic_list_camera1 = [...]
# transformation_matrix_dic_list_camera2 = [...]

T_cam1_to_qr_avg = calculate_camera_pose_from_qr_code_avg(transformation_matrix_dic_list_camera1)
T_cam2_to_qr_avg = calculate_camera_pose_from_qr_code_avg(transformation_matrix_dic_list_camera2)

print("Average Transformation Matrix from QR code to Camera 1:\n", T_cam1_to_qr_avg)
print("Average Transformation Matrix from QR code to Camera 2:\n", T_cam2_to_qr_avg)
