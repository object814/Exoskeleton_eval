import pybullet as p
import pybullet_data
import json
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

def load_transformation_matrices_from_json(file_path):
    with open(file_path, 'r') as f:
        matrices_list = json.load(f)
    return [np.array(matrix) for matrix in matrices_list]

def matrix_to_position_and_quaternion(transformation_matrix):
    """
    Convert a 4x4 transformation matrix to position and quaternion.
    
    Args:
        transformation_matrix: The 4x4 transformation matrix.
    Returns:
        The position (x, y, z) and quaternion (x, y, z, w).
    """
    position = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]
    # Convert rotation matrix to quaternion using SciPy
    rot = R.from_matrix(rotation_matrix)
    quaternion = rot.as_quat()  # Returns (x, y, z, w)
    return position, quaternion

def set_pose(body_id, transformation_matrix):
    position, quaternion = matrix_to_position_and_quaternion(transformation_matrix)
    p.resetBasePositionAndOrientation(body_id, position, quaternion)

def visualize_qr_code_movement(json_file_path, qr_code_path, visualize_axis=False, axis_length=0.1):
    matrices = load_transformation_matrices_from_json(json_file_path)
    
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load plane to represent QR code, adjusting its size to the real QR code size
    plane_id = p.loadURDF(qr_code_path)

    for matrix in matrices:
        if np.array_equal(matrix, np.identity(4)):
            continue
        set_pose(plane_id, matrix)

        if visualize_axis:
            # Get the position and orientation of the object
            position, orientation = p.getBasePositionAndOrientation(plane_id)

            # Convert orientation quaternion to rotation matrix
            rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

            # Compute rotated axes based on the orientation
            x_axis = np.dot(rotation_matrix, [axis_length, 0, 0]) + position
            y_axis = np.dot(rotation_matrix, [0, axis_length, 0]) + position
            z_axis = np.dot(rotation_matrix, [0, 0, axis_length]) + position
            
            # Draw coordinate axes at the object's origin with orientation
            p.addUserDebugLine(position, x_axis, [1, 0, 0], 5)  # X-axis (red)
            p.addUserDebugLine(position, y_axis, [0, 1, 0], 5)  # Y-axis (green)
            p.addUserDebugLine(position, z_axis, [0, 0, 1], 5)  # Z-axis (blue)

        input("Press Enter to continue...")
        time.sleep(1/10)  # Adjust sleep time to match desired visualization speed
        
    p.disconnect()

# Example usage:
# visualize_qr_code_movement('path_to_saved_matrices.json', qr_code_real_size)
if __name__ == '__main__':
    visualize_qr_code_movement('data/transformation_matrices.json', "assets/qr_code.urdf", True)