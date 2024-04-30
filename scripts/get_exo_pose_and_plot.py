import numpy as np
import pybullet as p
import json
import time
import os
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import threading
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.opt_exo_pose import set_frame_pose_with_qr_code_pose

def transform_A2B(A, B):
    """
    Calculate the transformation matrix C such that A * C = B.

    Args:
    - A (numpy.ndarray): Transformation matrix A.
    - B (numpy.ndarray): Transformation matrix B.

    Returns:
    - numpy.ndarray: The transformation matrix C.
    """
    # Calculate C using the matrix inverse of A and multiplication by B
    A_inv = np.linalg.inv(A)
    C = A_inv @ B
    return C

def set_trunk_frame_pose(trunk_pose, exo_id, trunk_id, client_id):
    """
    Sets the pose of the trunk frame based on the given transformation matrix and basis matrix.
    Basis matrix is pre-defined here to represent the relation between trunk frame and the EXO base frame.

    Args:
        transformation_matrix (numpy.ndarray): The transformation matrix representing the initial pose of the trunk frame.
        object_id (int): The ID of the object whose pose needs to be set.
        client_id (PyBulletClient): The PyBullet client object used to interact with the physics simulation.

    Returns:
        None
    """
    # Extract target trunk pose (calculated from QR code pose)
    target_trunk_position = trunk_pose[:3, 3]
    target_trunk_orientation_matrix = trunk_pose[:3, :3]
    target_trunk_orientation_quat = R.from_matrix(target_trunk_orientation_matrix).as_quat()

    # Set the base link position and orientation
    p.resetBasePositionAndOrientation(exo_id, target_trunk_position, target_trunk_orientation_quat, physicsClientId=client_id)
    
    # Read the current trunk pose
    link_state = p.getLinkState(exo_id, trunk_id, physicsClientId=client_id)
    trunk_position = link_state[4]  # World link frame position
    trunk_orientation_quat = link_state[5]  # World link frame orientation (quaternion)

    # Calculate the offset between the target and current trunk pose
    offset = target_trunk_position - np.array(trunk_position)

    # Reset the base link position and orientation
    base_position = target_trunk_position + offset
    p.resetBasePositionAndOrientation(exo_id, base_position, target_trunk_orientation_quat, physicsClientId=client_id)

def set_exo_pose(joint_angles, exo_id, client_id):
    """
    Sets the pose of the exoskeleton based on the given joint angles.

    Args:
    - joint_angles (list): The joint angles to set for the exoskeleton.
    - exo_id (int): The ID of the exoskeleton in the simulation.
    - client_id (int): The client ID of the PyBullet session.

    Returns:
    - None
    """
    for joint_index in range(len(joint_angles)):
        p.resetJointState(exo_id, joint_index, joint_angles[joint_index], physicsClientId=client_id)

    # for i, angle in enumerate(joint_angles):
    #     p.resetJointState(exo_id, i, angle, physicsClientId=client_id)

def get_kinematic_chain(exo_id, link_id, physicsClientId):
    kinematic_chain = []
    kinematic_chain_names = []
    while link_id != -1:  # -1 indicates the base
        joint_info = p.getJointInfo(exo_id, link_id, physicsClientId=physicsClientId)
        if joint_info[2] != p.JOINT_FIXED:  # Exclude fixed joints
            kinematic_chain.append(joint_info[0])
            kinematic_chain_names.append(joint_info[1].decode('utf-8'))
        link_id = joint_info[16]  # Parent link index
    # print(f"Kinematic chain: {kinematic_chain_names}")
    return kinematic_chain[::-1], kinematic_chain_names[::-1] # Return reversed list to start from the base

def read_frame_pose(exo_id, link_id, client_id):
    """
    Read the pose of the given link in the world frame in transformation matrix form.
    """
    link_state = p.getLinkState(exo_id, link_id, computeForwardKinematics=True, physicsClientId=client_id)
    link_position = link_state[4]  # World position
    link_orientation = link_state[5]  # World orientation
    link_pose = np.eye(4)
    link_pose[:3, 3] = link_position
    link_pose[:3, :3] = R.from_quat(link_orientation).as_matrix()
    return link_pose

def get_joint_angles_and_set(link_pose, exo_id, link_id, client_id):
    """
    Calculate the joint angles using inverse kinematics and set the robot to the target pose.
    
    Args:
    - link_pose (list of list): The target 4x4 transformation matrix of the link with respect to the world frame
    - exo_id (int): The ID of the robot in the simulation
    - link_id (int): The link ID for which the target pose needs to be achieved
    - client_id (int): The client ID of the PyBullet session

    Returns:
    - list: The joint angles to reach the target pose
    """
    # Extract position and rotation matrix from the transformation matrix
    position = np.array(link_pose)[:3, 3]
    rotation_matrix = np.array(link_pose)[:3, :3]

    # Convert the rotation matrix to a quaternion
    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()  # Returns [x, y, z, w]

    # Read the current joint angles
    current_joint_angles = read_joint_angles(exo_id, client_id)

    # Calculate the inverse kinematics
    joint_angles = p.calculateInverseKinematics(exo_id, link_id, targetPosition=position, targetOrientation=quaternion, physicsClientId=client_id)
    joint_angles = list(joint_angles)
    # Add 0 at joint_angles[2, 5, 6] because they are fixed joints
    joint_angles.insert(2, 0)
    joint_angles.insert(5, 0)
    joint_angles.insert(6, 0)

    # Read the kinematic chain of the link
    kinematic_chain, kinematic_chain_names = get_kinematic_chain(exo_id, link_id, client_id)

    # Apply the related joint angles to the robotnum_joints = p.getNumJoints(exo_id, physicsClientId=client_id)
    for joint_index in range(len(joint_angles)):
        if joint_index not in kinematic_chain:
            joint_angles[joint_index] = current_joint_angles[joint_index]
    for joint_index in range(len(joint_angles)):
        p.resetJointState(exo_id, joint_index, joint_angles[joint_index], physicsClientId=client_id)

    # Return the calculated joint angles
    return joint_angles

def read_joint_angles(exo_id, client_id):
    """
    Read the joint angles of the exoskeleton.
    """
    joint_angles = []
    for i in range(p.getNumJoints(exo_id, physicsClientId=client_id)):
        joint_info = p.getJointInfo(exo_id, i, physicsClientId=client_id)
        joint_angles.append(p.getJointState(exo_id, i, physicsClientId=client_id)[0])
    return joint_angles

def create_frame():
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axis_ids = []
    for color in colors:
        id = p.addUserDebugLine([0, 0, 0], [0, 0, 0], lineColorRGB=color, lineWidth=4)
        axis_ids.append(id)
    return axis_ids

def update_frame(axis_ids, transformation_matrix, width=1, length=0.1):
    origin = transformation_matrix[:3, 3]
    for i, axis_id in enumerate(axis_ids):
        direction = transformation_matrix[:3, i]
        # change the length of the direction to length
        direction = direction / np.linalg.norm(direction) * length
        p.addUserDebugLine(origin, origin + direction, lineColorRGB=[i == j for j in range(3)], replaceItemUniqueId=axis_id, lineWidth=width)

def read_trunk_inclination(exo_id, trunk_id, client_id):
    """
    Read the inclination of the trunk frame with respect to the world frame.
    The inclination is the angle between the z-axis of the trunk frame and the z-axis of the world frame.
    Range between 0 and pi.
    """
    trunk_pose = read_frame_pose(exo_id, trunk_id, client_id)
    trunk_z_axis = trunk_pose[:3, 2]
    world_z_axis = np.array([0, 0, 1])
    inclination = np.arccos(np.dot(trunk_z_axis, world_z_axis))
    return inclination

def main(qr_pose_path,
         exo_urdf_path,
         exo_frame_names = ["trunk", "rightThigh", "leftThigh"],
         qr_code_labels = ["2", "4", "3"],
         qr_code_target_axis = [None, "-y", "y"]):
    """
    Main function for exoskeleton pose estimation with QR code pose

    Pipeline:
    1. Read QR code pose from npy file
        structure of QR code pose data:
            {
                "1": [list of transformation matrices for QR code with label "1"]
                "2": [list of transformation matrices for QR code with label "2"]
                ...
            }
        * identity matrices in list means QR code pose in that step is not detected
        * trunk (base) frame corresponding QR code pose is always detected
    2. Initialize Pybullet simulation
    3. Declare the relationship between exoskeleton frame name and QR code label
    4. Initialize exoskeleton pose with QR poses in the first frame
        for trunk (base) frame: set the pose of trunk frame to the corresponding QR code pose
        for other frames: do optimization with the corresponding QR code pose
    5. Turn QR code pose data into relative transformation matrices
        calculate relative transformation matrices of QR code with respect to previous step, apply to the exoskeleton frame
        * undetected QR code frames:
            calculate the relative transformation matrix from the undetected frame to the trunk (base) frame in last step
            multiply the relative transformation matrix with the current step trunk (base) pose to get current step undetected frame pose
    6. Do inverse kinematics to find joint angles at every step
    7. Set the pose of exoskeleton in Pybullet simulation
    8. Save the joint angles to npy file
    """

    ###### Read QR code pose from npy file ######
    while True:
        try:
            qr_pose_data = np.load(qr_pose_path, allow_pickle=True).item()
            break
        except FileNotFoundError:
            print(f"File path \033[92m{qr_pose_path}\033[0m not found. Please provide a valid path.")
            return 0
            
    # cut the QR code pose data starting from frame 550
    for qr_label in qr_pose_data.keys():
        qr_pose_data[qr_label] = qr_pose_data[qr_label][550:]

    # get the total number of steps with any key
    total_step = len(qr_pose_data[list(qr_pose_data.keys())[0]])
    
    ###### Initialize Pybullet simulation and definitions ######
    p_client = p.connect(p.GUI) # visualization client
    p.setGravity(0, 0, -9.81, physicsClientId=p_client)
    p.setTimeStep(1/240, physicsClientId=p_client)

    # load URDF
    initialBasePosition = [0, 0, 0]
    initialBaseOrientation = p.getQuaternionFromEuler([0, 0, 0])
    exo_id = p.loadURDF(exo_urdf_path, basePosition=initialBasePosition, baseOrientation=initialBaseOrientation, useFixedBase=True, physicsClientId=p_client)
    base_link_name = p.getBodyInfo(exo_id, physicsClientId=p_client)[0].decode('utf-8')
    print("Base Link Name:", base_link_name)


    # find link ids by names
    links_to_visualize = ['trunk', 'leftThigh', 'rightThigh']
    num_joints = p.getNumJoints(exo_id, physicsClientId=p_client)
    link_ids = {}
    for i in range(num_joints):
        link_info = p.getJointInfo(exo_id, i, physicsClientId=p_client)
        link_name = link_info[12].decode('utf-8')
        if link_name in links_to_visualize:
            link_ids[link_name] = i
            
    print("Joint Names and their Indices:")
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(exo_id, joint_index, physicsClientId=p_client)
        joint_name = joint_info[1].decode('utf-8')
        print(f"Index: {joint_index}, Name: {joint_name}")
    
    # create visualization frames
    qr_trunk = create_frame()
    qr_leftThigh = create_frame()
    qr_rightThigh = create_frame()
    frame_trunk = create_frame()
    frame_leftThigh = create_frame()
    frame_rightThigh = create_frame()

    # find target axis by names
    axis_names = {}
    for exo_frame_name, target_axis in zip(exo_frame_names, qr_code_target_axis):
        if target_axis is None:
            pass
        else:
            axis_names[exo_frame_name] = target_axis

    # list to store joint angles at each step
    joint_angles_data = [read_joint_angles(exo_id, p_client)]

    ###### Declare the relationship between exoskeleton frame name and QR code label ######
    exo_frame_pose = {}
    for qr_label, exo_frame_name in zip(qr_code_labels, exo_frame_names):
        if qr_label not in qr_pose_data.keys():
            print(f"QR code label \033[92m{qr_label}\033[0m is not found in QR code pose data.")
            return 0
        qr_pose_data[exo_frame_name] = qr_pose_data.pop(qr_label) # rename the key of QR code pose data
        exo_frame_pose[exo_frame_name] = [] # initialize exoskeleton frame pose

    ###### Change the QR code frame into URDF virtual frame
    trunk_QR2Virtual = np.array([
        [ 0, -1, 0, 0],
        [ 0, 0, 1, 0],
        [ -1, 0, 0, 0],
        [ 0, 0, 0, 1]
    ])

    leftThigh_QR2Virtual = np.array([
        [ -1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]
    ])
    rightThigh_QR2Virtual = np.array([
        [ 1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, -1, 0, 0],
        [ 0, 0, 0, 1]
    ])

    for i in range(total_step):
        for exo_frame_name in exo_frame_names:
            if exo_frame_name == "trunk":
                exo_frame_pose[exo_frame_name].append(qr_pose_data[exo_frame_name][i] @ trunk_QR2Virtual)
            elif exo_frame_name == "leftThigh":
                exo_frame_pose[exo_frame_name].append(qr_pose_data[exo_frame_name][i] @ leftThigh_QR2Virtual)
            elif exo_frame_name == "rightThigh":
                exo_frame_pose[exo_frame_name].append(qr_pose_data[exo_frame_name][i] @ rightThigh_QR2Virtual)
            else:
                exo_frame_pose[exo_frame_name].append(qr_pose_data[exo_frame_name][i])

    # ###### Turn QR code pose data into relative transformation matrices ######
    # # do the base frame first
    # for i in range(1, total_step):
    #     previous2current_qr = transform_A2B(qr_pose_data["trunk"][i-1], qr_pose_data["trunk"][i])
    #     exo_frame_pose["trunk"].append(exo_frame_pose["trunk"][-1] @ previous2current_qr)
    # # do the other frames
    # for i in range(1, total_step):
    #     for exo_frame_name in exo_frame_names:
    #         if exo_frame_name == "trunk":
    #             continue
    #         else:
    #             if np.array_equal(qr_pose_data[exo_frame_name], np.identity(4)): # undetected QR code frame
    #                 base2frame = transform_A2B(exo_frame_pose["trunk"][i-1], exo_frame_pose[exo_frame_name][i-1])
    #                 current_frame_pose = exo_frame_pose["trunk"][i] @ base2frame # same pose as previous step with respect to base frame
    #                 previous2current_frame = transform_A2B(exo_frame_pose[exo_frame_name][i-1], current_frame_pose)
    #                 current_qr_pose = qr_pose_data[exo_frame_name][i-1] @ previous2current_frame # update the qr pose for upcoming calculation
    #                 qr_pose_data[exo_frame_name][i] = current_qr_pose
    #                 exo_frame_pose[exo_frame_name].append(current_frame_pose)
    #             else:
    #                 previous2current_qr = transform_A2B(qr_pose_data[exo_frame_name][i-1], qr_pose_data[exo_frame_name][i])
    #                 exo_frame_pose[exo_frame_name].append(exo_frame_pose[exo_frame_name][-1] @ previous2current_qr)


    ###### Do inverse kinematics to find joint angles at every step ######
    trunk_inclination_data = []
    for i in range(total_step):
        set_trunk_frame_pose(exo_frame_pose["trunk"][i], exo_id, link_ids["trunk"], p_client)
        set_exo_pose(joint_angles_data[-1], exo_id, p_client)
        for exo_frame_name in exo_frame_names:
            if exo_frame_name == "trunk":
                trunk_inclination = read_trunk_inclination(exo_id, link_ids["trunk"], p_client)
            else:
                joint_angles = get_joint_angles_and_set(exo_frame_pose[exo_frame_name][i], exo_id, link_ids[exo_frame_name], p_client)
        trunk_inclination_data.append(trunk_inclination)
        joint_angles_data.append(joint_angles)

    joint_angles_data = np.array(joint_angles_data)
    trunk_inclination_data = np.array(trunk_inclination_data)

    total_step = len(joint_angles_data)
    num_joints = joint_angles_data.shape[1]

    # Setup Matplotlib figures for real-time updating
    fig, axs = plt.subplots(2, 2)
    lines = []
    for i in range(2):
        for j in range(2):
            line, = axs[i, j].plot([], [])
            axs[i, j].set_xlim(0, total_step)
            axs[i, j].set_ylim(np.min(joint_angles_data[:, i*2+j]), np.max(joint_angles_data[:, i*2+j]))
            lines.append(line)

    # Set titles and labels
    axs[0, 0].set_title("Left Thigh Joint 1 Angle")
    axs[0, 1].set_title("Left Thigh Joint 2 Angle")
    axs[1, 0].set_title("Right Thigh Joint 1 Angle")
    axs[1, 1].set_title("Right Thigh Joint 2 Angle")
    for ax in axs.flat:
        ax.set(xlabel='TimeStep', ylabel='Angle (rad)')

    # Prepare the trunk inclination plot
    fig_trunk, ax_trunk = plt.subplots()
    line_trunk, = ax_trunk.plot([], [])
    ax_trunk.set_xlim(0, total_step)
    ax_trunk.set_ylim(np.min(trunk_inclination_data), np.max(trunk_inclination_data))
    ax_trunk.set_title("Trunk Inclination")
    ax_trunk.set_xlabel("Time")
    ax_trunk.set_ylabel("Inclination (rad)")

    plt.show(block=False)

    # Pybullet visualization loop with Matplotlib updates
    input("start visualizing")
    for i in range(total_step):
        set_trunk_frame_pose(exo_frame_pose["trunk"][i], exo_id, link_ids["trunk"], p_client)
        set_exo_pose(joint_angles_data[i], exo_id, p_client)

        # Update frames in Pybullet
        update_frame(qr_trunk, exo_frame_pose["trunk"][i])
        update_frame(qr_leftThigh, exo_frame_pose["leftThigh"][i])
        update_frame(qr_rightThigh, exo_frame_pose["rightThigh"][i])
        update_frame(frame_trunk, read_frame_pose(exo_id, link_ids["trunk"], p_client), width=1.5)
        update_frame(frame_leftThigh, read_frame_pose(exo_id, link_ids["leftThigh"], p_client), width=1.5)
        update_frame(frame_rightThigh, read_frame_pose(exo_id, link_ids["rightThigh"], p_client), width=1.5)
        
        # Update Matplotlib plots
        for j, line in enumerate(lines):
            if j == 2 or j == 3:
                line.set_data(range(i+1), joint_angles_data[:i+1, j+1])
            else:
                line.set_data(range(i+1), joint_angles_data[:i+1, j])
            axs[j//2, j%2].draw_artist(axs[j//2, j%2].patch)
            axs[j//2, j%2].draw_artist(line)
        
        line_trunk.set_data(range(i+1), trunk_inclination_data[:i+1])
        ax_trunk.draw_artist(ax_trunk.patch)
        ax_trunk.draw_artist(line_trunk)

        plt.pause(0.05)  # Short pause to update plots
        print(f"trunk inclination: {trunk_inclination_data[i]}")

    p.disconnect(p_client)

if __name__ == "__main__":
    main(qr_pose_path="data/0422_test/unified_poses.npy",
         exo_urdf_path="data/exo_model/urdf/exo_w_virtual_frame.urdf",
         exo_frame_names=["trunk", "leftThigh", "rightThigh"],
         qr_code_labels=["2", "3", "4"],
         qr_code_target_axis=[None, "y", "-y"])