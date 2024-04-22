import pybullet as p
import pybullet_data
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

def draw_frame(object_id, link_id, frame_length=0.1, lifetime=0):
    # Draws XYZ frame for a given object and link ID.
    link_state = p.getLinkState(object_id, link_id, computeForwardKinematics=True)
    link_position = link_state[4]  # World position
    link_orientation = link_state[5]  # World orientation

    # Axes directions in local frame
    x_axis = [frame_length, 0, 0]
    y_axis = [0, frame_length, 0]
    z_axis = [0, 0, frame_length]

    # Transform axes to world frame
    x_axis_world = p.multiplyTransforms(link_position, link_orientation, x_axis, [0, 0, 0, 1])[0]
    y_axis_world = p.multiplyTransforms(link_position, link_orientation, y_axis, [0, 0, 0, 1])[0]
    z_axis_world = p.multiplyTransforms(link_position, link_orientation, z_axis, [0, 0, 0, 1])[0]

    # Draw the axes in world frame
    p.addUserDebugLine(link_position, x_axis_world, [1, 0, 0], lifeTime=lifetime, lineWidth=2)
    p.addUserDebugLine(link_position, y_axis_world, [0, 1, 0], lifeTime=lifetime, lineWidth=2)
    p.addUserDebugLine(link_position, z_axis_world, [0, 0, 1], lifeTime=lifetime, lineWidth=2)

def get_initial_direction_and_position(object_id, link_id):
    # Retrieve link state to get initial position and orientation
    link_state = p.getLinkState(object_id, link_id, computeForwardKinematics=True)
    link_position = link_state[4]  # World position of the link
    link_orientation = link_state[5]  # World orientation of the link as a quaternion

    # Convert link orientation from quaternion to rotation matrix
    rotation_matrix = R.from_quat(link_orientation).as_matrix()

    pose = np.eye(4)
    pose[:3, 3] = np.array(link_position)
    pose[:3, :3] = np.array(rotation_matrix)

    return pose

def set_trunk_frame_pose(transformation_matrix, object_id, pb_client):
    """
    Sets the pose of the trunk frame based on the given transformation matrix and basis matrix.

    Args:
        transformation_matrix (numpy.ndarray): The transformation matrix representing the initial pose of the trunk frame.
        object_id (int): The ID of the object whose pose needs to be set.
        pb_client (PyBulletClient): The PyBullet client object used to interact with the physics simulation.

    Returns:
        None
    """
    # Extract initial position and orientation from the transformation matrix
    initial_position = transformation_matrix[:3, 3]
    initial_orientation_matrix = transformation_matrix[:3, :3]
    
    # Convert initial orientation matrix to quaternion using SciPy
    initial_orientation_quat = R.from_matrix(initial_orientation_matrix).as_quat()
    
    # Apply basis transformation
    '''
    The rotation in basis transformation aligns the qr code frame definition with the pybullet trunk frame definition
    The translation in basis transformation moves the trunk frame origin to the base frame origin
    '''
    basis_matrix = np.array([
        [ 0, -1,  0, 0.19827],
        [ 0,  0,  1, -0.0004412],
        [-1,  0,  0, 0.18652],
        [ 0,  0,  0, 1]
    ])
    basis_position = basis_matrix[:3, 3]
    basis_orientation_matrix = basis_matrix[:3, :3]
    
    # Convert basis orientation matrix to quaternion using SciPy
    basis_orientation_quat = R.from_matrix(basis_orientation_matrix).as_quat()
    
    # Calculate the final position by adding the basis position
    final_position = initial_position + basis_position
    
    # For orientation, quaternion multiplication is needed to apply the rotation
    # SciPy can handle the quaternion multiplication
    final_orientation = R.from_quat(initial_orientation_quat) * R.from_quat(basis_orientation_quat)
    final_orientation_quat = final_orientation.as_quat()
    
    # Use PyBullet to set the object's pose
    pb_client.resetBasePositionAndOrientation(object_id, final_position, final_orientation_quat)


# def calculate_and_transform(direction_ref, direction_target, direction_new_ref):
#     """
#     Calculates and transforms a direction vector from a reference direction to a target direction.

#     Args:
#     - direction_ref: numpy array representing the reference direction
#     - direction_target: numpy array representing the target direction
#     - direction_new_ref: numpy array representing the direction to be transformed

#     Returns:
#     - direction_transformed: numpy array representing the transformed direction

#     The function normalizes the input vectors to ensure they represent directions.
#     It then calculates the rotation from the reference direction to the target direction.
#     Finally, it applies the same rotation to the direction_new_ref vector and returns the transformed direction.
#     """
#     # Normalize vectors to ensure they represent directions
#     direction_ref = direction_ref / np.linalg.norm(direction_ref)
#     direction_target = direction_target / np.linalg.norm(direction_target)
#     direction_new_ref = direction_new_ref / np.linalg.norm(direction_new_ref)

#     # Calculate rotation from direction_ref to direction_target
#     axis_rotate = np.cross(direction_ref, direction_target)
#     angle_rotate = np.arccos(np.dot(direction_ref, direction_target))

#     # Construct rotation matrix to transform direction_ref to direction_target
#     R_rotate = rotation_matrix_from_axis_angle(axis_rotate, angle_rotate)

#     # Apply the same rotation to direction_new_ref
#     direction_transformed = np.dot(R_rotate, direction_new_ref)

#     return direction_transformed

# def rotation_matrix_from_axis_angle(axis, angle):
#     """
#     Compute a rotation matrix from an axis and an angle.

#     Args:
#     axis (numpy.ndarray): The axis of rotation.
#     angle (float): The angle of rotation in radians.

#     Returns:
#     numpy.ndarray: The rotation matrix.

#     """
#     axis = axis / np.linalg.norm(axis)
#     a = np.cos(angle / 2)
#     b, c, d = -axis * np.sin(angle / 2)
    
#     return np.array([[a*a + b*b - c*c - d*d, 2 * (b*c - a*d), 2 * (b*d + a*c)],
#                      [2 * (b*c + a*d), a*a + c*c - b*b - d*d, 2 * (c*d - a*b)],
#                      [2 * (b*d - a*c), 2 * (c*d + a*b), a*a + d*d - b*b - c*c]])

def main(urdf_path, poses_path):
    # Read poses from file
    qr_poses = np.load(poses_path, allow_pickle=True).item()

    # Change the keys in poses_vision to match the link names in the URDF
    '''
    2: trunk
    3: leftThigh
    4: rightThigh
    '''
    poses_vision = {}
    poses_vision["trunk"] = qr_poses.pop("2")
    poses_vision["rightThigh"] = poses_vision.pop("3")
    poses_vision["leftThigh"] = poses_vision.pop("4")


    # Connect to PyBullet
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    # Load exoskeleton URDF
    initialBasePosition = [0, 0, 0]
    initialBaseOrientation = p.getQuaternionFromEuler([0, 0, 0])
    exo_id = p.loadURDF(urdf_path, basePosition=initialBasePosition, baseOrientation=initialBaseOrientation, useFixedBase=True)

    # Find the link IDs by names
    links_to_visualize = ['trunk', 'leftThigh', 'rightThigh']
    num_joints = p.getNumJoints(exo_id)
    link_ids = {}

    for i in range(num_joints):
        link_info = p.getJointInfo(exo_id, i)
        link_name = link_info[12].decode('utf-8')
        if link_name in links_to_visualize:
            link_ids[link_name] = i

    # Set initial pose for pybullet links
    # Trunk link
    set_trunk_frame_pose(poses_vision["trunk"][0], exo_id, p) # set the trunk to be at [0,0,1.5]
    # Left and Right links
    # set_thigh_frame_pose("virtual_link_left", poses_vision["leftThigh"][0][0], exo_id, 'y', p)
    # set_trunk_frame_pose("virtual_link_right", poses_vision["rightThigh"][0][0], exo_id, '-y', p)

    # Find the link initial pose in pybullet
    poses_pybullet = {
        "trunk": [],
        "leftThigh": [],
        "rightThigh": []
    }
    for link_name in links_to_visualize:
        link_id = link_ids[link_name]
        link_pose = get_initial_direction_and_position(exo_id, link_id)

        if link_name == 'trunk':
            # For trunk, interested in -x direction
            poses_pybullet["trunk"].append(link_pose)
        elif link_name == 'leftThigh':
            # For left, interested in y direction
            poses_pybullet["leftThigh"].append(link_pose)
        elif link_name == 'rightThigh':
            # For right, interested in -y direction
            poses_pybullet["rightThigh"].append(link_pose)
    
    # Transform the positions in poses_vision into poses_pybullet
    for link_name, poses in poses_vision.items():
        for i in range(len(poses)):
            if i == 0:
                continue
            else:
                pose_pybullet = np.eye(4)
                relative_pos = np.array(poses[i][:3, 3]) - np.array(poses[i-1][:3, 3]) # calculate the relative position
                pybullet_pos = np.array(poses_pybullet[link_name][-1][:3, 3]) # get the previous pybullet position
                pose_pybullet[:3, 3] = pybullet_pos + relative_pos
                pose_pybullet[:3, :3] = np.array(poses[i][:3, :3])
                poses_pybullet[link_name].append(pose_pybullet)


    # for i in range(10):
    #     print(poses_vision["trunk"][i])
    # for i in range(10):
    #     print(poses_pybullet["trunk"][i])

    # Step simulation
    for step in range(1, len(poses_pybullet["trunk"])):
        # Set the pose for each link
        # for link_name in links_to_visualize:
            # pose = poses_pybullet[link_name][step]
            # set_trunk_frame_pose(pose, exo_id, p)
        pose = poses_pybullet["trunk"][step]
        set_trunk_frame_pose(pose, exo_id, p)
        input("Press Enter to continue...")

    # Start simulation
    # while True:
    #     p.stepSimulation()
    #     for link_name, link_id in link_ids.items():
    #         draw_frame(exo_id, link_id)
    #     time.sleep(0.01)  # Time step for simulation
    #     input("Press Enter to continue...")
    
    p.disconnect()

if __name__ == '__main__':
    main('data/exo_model/urdf/exo_w_virtual_frame.urdf', 'data/0418_test_2/unified_poses.npy')
