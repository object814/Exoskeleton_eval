import pybullet as p
import pybullet_data
import numpy as np
import time

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
    rotation_matrix = np.array(p.getMatrixFromQuaternion(link_orientation)).reshape((3, 3))

    # Extract direction vectors in the link's local frame
    x_direction = rotation_matrix[:, 0]  # First column
    y_direction = rotation_matrix[:, 1]  # Second column

    return x_direction, y_direction, link_position

def set_trunk_frame_pose(direction, position, body_id, p_id):
    """
    Set the pose of the trunk frame in PyBullet.

    Args:
        direction (list): The target direction vector for the x-axis of the trunk frame.
        position (list): The target position of the trunk frame.
        body_id (int): The ID of the body in PyBullet.
        p_id (object): The PyBullet physics client object.

    Returns:
        None
    """
    # Fetch the current orientation of the base link in quaternion form
    _, current_quat = p_id.getBasePositionAndOrientation(body_id)
    
    # Convert the current quaternion to a direction vector for the x-axis
    current_dir_matrix = p_id.getMatrixFromQuaternion(current_quat)
    current_dir_x = np.array([current_dir_matrix[0], current_dir_matrix[3], current_dir_matrix[6]])
    
    # Normalize the target direction vector
    direction = np.array(direction)
    direction_norm = direction / np.linalg.norm(direction)
    
    # Determine if the current direction or its opposite is closer to the target direction
    if np.dot(current_dir_x, direction_norm) < np.dot(current_dir_x, -direction_norm):
        direction_norm = -direction_norm  # Use the opposite direction for alignment
    
    # Calculate the rotation axis and angle for the quaternion
    align_axis = -np.array([1, 0, 0])
    rotation_axis = np.cross(align_axis, direction_norm)
    if np.linalg.norm(rotation_axis) < 1e-6:
        rotation_axis = None
        angle = 0
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.dot(align_axis, direction_norm))
    
    # Calculate quaternion for the rotation
    if rotation_axis is not None:
        qx = rotation_axis[0] * np.sin(angle / 2)
        qy = rotation_axis[1] * np.sin(angle / 2)
        qz = rotation_axis[2] * np.sin(angle / 2)
        qw = np.cos(angle / 2)
        quaternion = [qx, qy, qz, qw]
    else:
        quaternion = [0, 0, 0, 1]  # No rotation, identity quaternion

    # Adjust the position as per your existing code
    position_adjusted = np.array(position) - np.array([-0.19827, 0.0004412, -0.18652])

    # Update the base link pose in PyBullet
    p_id.resetBasePositionAndOrientation(body_id, position_adjusted, quaternion)

def main(urdf_path):
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    # Load exoskeleton URDF
    initialBasePosition = [0, 0, 0]
    initialBaseOrientation = p.getQuaternionFromEuler([0, 0, 0])
    exo_id = p.loadURDF(urdf_path, basePosition=initialBasePosition, baseOrientation=initialBaseOrientation, useFixedBase=True)

    # Find the link IDs by names
    links_to_visualize = ['trunk', 'rightThigh', 'leftThigh']
    num_joints = p.getNumJoints(exo_id)
    link_ids = {}

    for i in range(num_joints):
        link_info = p.getJointInfo(exo_id, i)
        link_name = link_info[12].decode('utf-8')
        if link_name in links_to_visualize:
            link_ids[link_name] = i

    # Start simulation
    while True:
        p.stepSimulation()
        for link_name, link_id in link_ids.items():
            draw_frame(exo_id, link_id)
        time.sleep(0.01)  # Time step for simulation
        input("Press Enter to continue...")
    
    p.disconnect()

if __name__ == '__main__':
    main('assets/exo_augmented/urdf/back_exo.urdf')
