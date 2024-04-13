import pybullet as p
import pybullet_data
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import random

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

def cost_function(robot_id, link_index, target_dir, current_axis_name):
    current_dir = get_current_axis(robot_id, link_index, current_axis_name)
    cost = 1 - np.dot(current_dir, target_dir)
    return cost

def get_kinematic_chain(robot_id, link_index):
    """ Return a list of joint indices that are part of the kinematic chain leading to the specified link. """
    kinematic_chain = []
    while link_index != -1:  # -1 indicates the base
        joint_info = p.getJointInfo(robot_id, link_index)
        if joint_info[2] != p.JOINT_FIXED:  # Exclude fixed joints
            kinematic_chain.append(joint_info[0])
        link_index = joint_info[16]  # Parent link index
    return kinematic_chain[::-1]  # Return reversed list to start from the base

def optimize_joints(robot_id, link_index, axis, iterations, target_dir):
    kinematic_chain = get_kinematic_chain(robot_id, link_index)
    target_dir = np.array(target_dir) / np.linalg.norm(target_dir)
    learning_rate = 0.005  # Reduced learning rate
    damping = 0.1  # Damping factor
    gradient_clip = 0.05  # Maximum gradient step

    for _ in range(iterations):
        gradients = np.zeros(len(kinematic_chain))

        for idx, joint_index in enumerate(kinematic_chain):
            original_angle = p.getJointState(robot_id, joint_index)[0]
            p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=original_angle + 0.01)
            p.stepSimulation()
            cost_perturbed = cost_function(robot_id, link_index, target_dir, axis)

            p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=original_angle)
            p.stepSimulation()
            current_cost = cost_function(robot_id, link_index, target_dir, axis)

            gradient = (cost_perturbed - current_cost) / 0.01
            gradients[idx] = gradient

        # Apply gradient clipping and damping
        gradients = np.clip(gradients, -gradient_clip, gradient_clip)
        for idx, joint_index in enumerate(kinematic_chain):
            original_angle = p.getJointState(robot_id, joint_index)[0]
            change = -learning_rate * gradients[idx] - damping * original_angle
            new_angle = original_angle + change
            p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=new_angle)

        p.stepSimulation()

    final_dir = get_current_axis(robot_id, link_index, axis)
    error = np.arccos(np.clip(np.dot(final_dir, target_dir), -1.0, 1.0))

    print(f"Optimization complete for {axis} axis on link index {link_index}.")
    print(f"Final direction: {final_dir}")
    print(f"Final error: {error/np.pi*180:.2f} degrees")

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

def randomize_joints_and_compute_direction(robot_id, link_index, axis):
    num_joints = p.getNumJoints(robot_id)
    joint_indices = range(num_joints)

    # Randomize joint angles
    for joint_index in joint_indices:
        joint_info = p.getJointInfo(robot_id, joint_index)
        if joint_info[2] == p.JOINT_REVOLUTE:
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if upper_limit > lower_limit:  # Check if the joint has limits
                random_angle = random.uniform(lower_limit, upper_limit)
                p.resetJointState(robot_id, joint_index, random_angle)

    # Compute the forward kinematics to get the direction of the axis
    p.stepSimulation()
    current_direction = get_current_axis(robot_id, link_index, axis)
    return current_direction

def get_current_axis(robot_id, link_index, axis):
    """Returns the current global direction vector of the specified local axis of the link."""
    _, link_orient = p.getLinkState(robot_id, link_index)[:2]
    axis_vectors = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1]),
        '-x': np.array([-1, 0, 0]),
        '-y': np.array([0, -1, 0]),
        '-z': np.array([0, 0, -1])
    }
    local_axis = axis_vectors[axis]
    current_axis = np.array(p.rotateVector(link_orient, local_axis))
    return current_axis


def main(urdf_path):
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)  # Assuming 240 Hz simulation frequency

    exo_id = p.loadURDF(urdf_path, useFixedBase=True)

    link_names = ['trunk', 'rightThigh', 'leftThigh']
    link_ids = {p.getJointInfo(exo_id, i)[12].decode('utf-8'): i for i in range(p.getNumJoints(exo_id)) if p.getJointInfo(exo_id, i)[12].decode('utf-8') in link_names}
    initial_positions = [p.getJointState(exo_id, i)[0] for i in range(p.getNumJoints(exo_id))]
    print(link_ids)
    
    # Declare the link name and axis for optimization
    link_name = input("Enter the link name to optimize (e.g., r, l): ")
    if link_name == 'r':
        link_name = 'rightThigh'
        axis = '-y'
    elif link_name == 'l':
        link_name = 'leftThigh'
        axis = 'y'

    # Check if the link name is valid, if not, ask the user to input again
    while link_name not in link_ids:
        print("Invalid link name. Please enter a valid link name.")
        link_name = input("Enter the link name to optimize (e.g., rightThigh, leftThigh): ")

    # Run simulation until the user confirms to capture the pose
    while True:
        p.stepSimulation()
        time.sleep(0.01)
        current_direction = get_current_axis(exo_id, link_ids[link_name], axis)
        print("Captured direction for optimization:", current_direction)
        # wait until the user presses 'y' to capture the pose
        keys = p.getKeyboardEvents()
        if ord('y') in keys and keys[ord('y')] & p.KEY_WAS_TRIGGERED:
            break

    current_direction = get_current_axis(exo_id, link_ids[link_name], axis)
    print("Captured direction for optimization:", current_direction)

    # Capture the current joint angles and direction for the specified link and axis
    target_joint_angles = [p.getJointState(exo_id, i)[0] for i in range(p.getNumJoints(exo_id))]

    # Reset the robot to initial positions
    for i in range(p.getNumJoints(exo_id)):
        p.resetJointState(exo_id, i, initial_positions[i])
    p.stepSimulation()  # One step to apply the reset states

    input("Press Enter to start the optimization process...")

    # Optimization
    for i in range(10): # optimize for ten times
        optimize_joints(exo_id, link_ids[link_name], axis, 1000, current_direction.tolist())
        input("Press Enter to continue...")
        for i in range(p.getNumJoints(exo_id)):
            p.resetJointState(exo_id, i, initial_positions[i])
        p.stepSimulation()  # One step to apply the reset states


    p.disconnect()

if __name__ == '__main__':
    main('data/exo_model/urdf/exo_w_virtual_frame.urdf')
