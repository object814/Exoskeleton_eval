import numpy as np
from scipy.spatial.transform import Rotation as R

def is_identity_matrix(T):
    """Checks if a transformation matrix is close to the identity matrix."""
    return np.allclose(T, np.eye(4), atol=1e-8)

def invert_transformation_matrix(T):
    """Inverts a transformation matrix."""
    R_inv = T[:3, :3].T
    t_inv = -R_inv @ T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def average_transformation_matrices(transformation_matrices):
    """Averages a list of transformation matrices, excluding identity matrices."""
    valid_quaternions = []
    valid_translations = []

    for T in transformation_matrices:
        if not is_identity_matrix(T):
            R_mat = T[:3, :3]
            t_vec = T[:3, 3]
            quaternion = R.from_matrix(R_mat).as_quat()
            valid_quaternions.append(quaternion)
            valid_translations.append(t_vec)
    
    if not valid_quaternions:
        raise ValueError("No valid transformation matrices found. All matrices are identity or no matrices provided.")

    # Average quaternions and translations
    mean_quaternion = np.mean(valid_quaternions, axis=0)
    mean_translation = np.mean(valid_translations, axis=0)

    # Convert the average quaternion back to a rotation matrix
    mean_rotation_matrix = R.from_quat(mean_quaternion).as_matrix()

    # Construct the average transformation matrix
    T_avg = np.eye(4)
    T_avg[:3, :3] = mean_rotation_matrix
    T_avg[:3, 3] = mean_translation

    return T_avg

def get_camera_pose_in_qr_frame(transformation_matrices):
    """Calculates the camera pose from averaging QR code transformation matrices."""
    T_avg_qr_in_cam = average_transformation_matrices(transformation_matrices)
    T_avg_cam_in_qr = invert_transformation_matrix(T_avg_qr_in_cam)
    return T_avg_cam_in_qr

# Example usage
if __name__ == "__main__":
    # Sample transformation matrices for demonstration purposes
    # In practice, replace these with your actual matrices
    transformation_matrices_for_qr = [
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),  # Identity matrix (example of QR code not detected)
        # Add your real transformation matrices here
    ]

    try:
        T_cam_to_qr_avg = get_camera_pose_in_qr_frame(transformation_matrices_for_qr)
        print("Average Transformation Matrix from Camera to QR code:\n", T_cam_to_qr_avg)
    except ValueError as e:
        print(e)
