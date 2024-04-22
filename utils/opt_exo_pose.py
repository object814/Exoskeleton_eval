import pybullet as p
import numpy as np
from scipy.optimize import minimize

def get_current_axis_dir(exo_id, link_id, axis, client_id):
    # Draws XYZ frame for a given object and link ID.
    link_state = p.getLinkState(exo_id, link_id, computeForwardKinematics=True, physicsClientId=client_id)
    link_position = link_state[4]  # World position
    link_orientation = link_state[5]  # World orientation

    # Axes directions in local frame
    if axis == 'y':
        axis = [0, 1, 0]
    if axis == '-y':
        axis = [0, -1, 0]

    # Transform axes to world frame
    current_dir = p.multiplyTransforms(link_position, link_orientation, axis, [0, 0, 0, 1])[0]

    # Normalize the direction vector
    current_dir = np.array(current_dir)
    current_dir /= np.linalg.norm(current_dir)
    # p.addUserDebugLine(link_position, current_dir, [1, 0, 0], lifeTime=0, lineWidth=5, physicsClientId=physicsClientId)

    return current_dir

def cost_function(joint_angles, target_opt_dir, opt_axis, opt_joints, exo_id, link_id, client_id):
    # Set joint angles
    for i, angle in enumerate(joint_angles):
        p.resetJointState(exo_id, opt_joints[i], angle, physicsClientId=client_id)
        # p.setJointMotorControl2(exo_id, opt_joints[i], p.POSITION_CONTROL, targetPosition=angle, physicsClientId=client_id)
    # p.stepSimulation(physicsClientId=client_id)

    # Get the direction of the optimization link
    opt_dir = get_current_axis_dir(exo_id, link_id, opt_axis, client_id)

    # Calculate cost
    cost = 1 - np.dot(opt_dir, target_opt_dir)
    cost = cost * 100

    return cost

def draw_axis(exo_id, link_id, axis, physicsClientId, frame_length=0.1, lifetime=0):
    # Draws XYZ frame for a given object and link ID.
    link_state = p.getLinkState(exo_id, link_id, computeForwardKinematics=True, physicsClientId=physicsClientId)
    link_position = link_state[4]  # World position
    link_orientation = link_state[5]  # World orientation

    # Axes directions in local frame
    if axis == 'y':
        axis = [0, frame_length, 0]
    if axis == '-y':
        axis = [0, -frame_length, 0]

    # Transform axes to world frame
    axis_world = p.multiplyTransforms(link_position, link_orientation, axis, [0, 0, 0, 1])[0]

    # Draw the axes in world frame
    p.addUserDebugLine(link_position, axis_world, [0, 1, 0], lifeTime=lifetime, lineWidth=2, physicsClientId=physicsClientId)

def get_kinematic_chain(exo_id, link_id, physicsClientId):
    kinematic_chain = []
    kinematic_chain_names = []
    while link_id != -1:  # -1 indicates the base
        joint_info = p.getJointInfo(exo_id, link_id, physicsClientId=physicsClientId)
        if joint_info[2] != p.JOINT_FIXED:  # Exclude fixed joints
            kinematic_chain.append(joint_info[0])
            kinematic_chain_names.append(joint_info[1].decode('utf-8'))
        link_id = joint_info[16]  # Parent link index
    kinematic_chain_names[::-1]
    print(f"Kinematic chain: {kinematic_chain_names}")
    return kinematic_chain[::-1]  # Return reversed list to start from the base

def optimize_joints(target_opt_dir, opt_axis, opt_joints, exo_id, link_id, client_id, iterations=100, learning_rate=0.01, threshold=0.01):
    print("##################")
    # Normalize the target direction vector
    target_opt_dir = np.array(target_opt_dir) / np.linalg.norm(target_opt_dir)

    # Initialize optimization variables
    learning_rate = learning_rate
    threshold = threshold

    for _ in range(iterations):
        gradients = np.zeros(len(opt_joints)) # Initialize gradients
        current_opt_joint_angles = []
        for joint_id in opt_joints:
            current_opt_joint_angles.append(p.getJointState(exo_id, joint_id, physicsClientId=client_id)[0])

        for i in range(len(opt_joints)):
            perturbed_opt_joint_angles = current_opt_joint_angles.copy()
            perturbed_opt_joint_angles[i] += 0.01
            perturbed_cost = cost_function(perturbed_opt_joint_angles, 
                                           target_opt_dir, 
                                           opt_axis, 
                                           opt_joints,
                                           exo_id, 
                                           link_id,
                                           client_id)
            
            current_cost = cost_function(current_opt_joint_angles,
                                         target_opt_dir,
                                         opt_axis,
                                         opt_joints,
                                         exo_id, 
                                         link_id,
                                         client_id)

            gradient = (perturbed_cost - current_cost) / 0.01
            gradients[i] = gradient

        # Update joint angles
        new_opt_joint_angles = current_opt_joint_angles - learning_rate * gradients
        # Calculate cost
        cost = cost_function(new_opt_joint_angles, target_opt_dir, opt_axis, opt_joints, exo_id, link_id, client_id)
        if cost < threshold:
            print(f"Optimization converged at iteration {_}")
            break

    # Calculate final error
    final_opt_joint_angles = []
    for joint_id in opt_joints:
            final_opt_joint_angles.append(p.getJointState(exo_id, joint_id, physicsClientId=client_id)[0])
    final_cost = cost_function(final_opt_joint_angles, target_opt_dir, opt_axis, opt_joints, exo_id, link_id, client_id)
    final_dir = get_current_axis_dir(exo_id, link_id, opt_axis, client_id)
    angle_radians = np.arccos(np.clip(np.dot(final_dir, target_opt_dir) / (np.linalg.norm(final_dir) * np.linalg.norm(target_opt_dir)), -1.0, 1.0))
    angle_degrees = angle_radians / np.pi * 180

    print(f"Final cost: {final_cost}")
    print(f"Final direction: {final_dir}")
    print(f"Final difference: {angle_degrees} degrees")
    print("##################")

    return final_opt_joint_angles

def set_frame_pose_with_qr_code_pose(qr_code_pose, exo_id, link_id, opt_axis, client_id):
    print("-----------------")
    print("Start Optimization...")

    # Find the kinematic chain for the optimization link
    kinematic_chain = get_kinematic_chain(exo_id, link_id, physicsClientId=client_id)
    opt_joints = kinematic_chain

    # Initial optimize variable value
    initial_target_opt_dir = get_current_axis_dir(exo_id, link_id, opt_axis, client_id)
    print(f'Initial target direction: {initial_target_opt_dir}')

    # Extract target optimize variable value
    target_opt_dir = qr_code_pose[0:3, 2]
    print(f'Target direction: {target_opt_dir}')

    # Optimization
    result = optimize_joints(target_opt_dir, opt_axis, opt_joints, exo_id, link_id, client_id)
    for i, angle in enumerate(result):
            p.resetJointState(exo_id, opt_joints[i], angle, physicsClientId=client_id)

    print("Exoskeleton set to the target direction")
    print("Optimization Complete")
    print("-----------------")

    # # draw the target direction
    # p.addUserDebugLine([0, 0, 0], target_opt_dir, [0, 1, 0], lineWidth=5, physicsClientId=vis_client) # green
    # # draw the current direction
    # current_dir = get_current_axis_dir(vis_exo_id, link_index[opt_link_name], opt_axis, physicsClientId=vis_client)
    # p.addUserDebugLine([0, 0, 0], current_dir, [1, 0, 0], lineWidth=5, physicsClientId=vis_client) # red

    # # Output error
    # input('Press "Enter" to exit.')
    # p.disconnect()