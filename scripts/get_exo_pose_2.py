import numpy as np
import pybullet as p
import json
import os
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

def set_trunk_frame_pose(qr_pose, exo_id, client_id):
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
    # Extract initial position and orientation from the transformation matrix
    qr_code_position = qr_pose[:3, 3]
    qr_code_orientation_matrix = qr_pose[:3, :3]
    qr_code_orientation_quat = R.from_matrix(qr_code_orientation_matrix).as_quat()
    
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
    basis_orientation_quat = R.from_matrix(basis_orientation_matrix).as_quat()
    
    # Calculate trunk frame pose
    trunk_position = qr_code_position + basis_position
    trunk_orientation = R.from_quat(qr_code_orientation_quat) * R.from_quat(basis_orientation_quat)
    trunk_orientation_quat = trunk_orientation.as_quat()
    
    # Set exo base frame pose to make trunk frame align with qr code frame
    p.resetBasePositionAndOrientation(exo_id, trunk_position, trunk_orientation_quat, physicsClientId=client_id)

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
    for i, angle in enumerate(joint_angles):
        p.resetJointState(exo_id, i, angle, physicsClientId=client_id)

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

    # Calculate the inverse kinematics
    joint_angles = p.calculateInverseKinematics(exo_id, link_id, targetPosition=position, targetOrientation=quaternion, physicsClientId=client_id)
    
    # Apply the joint angles to the robot
    for i, angle in enumerate(joint_angles):
        p.resetJointState(exo_id, i, angle, physicsClientId=client_id)

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

def main(qr_pose_path,
         exo_urdf_path,
         exo_frame_names = ["trunk", "leftThigh", "rightThigh"],
         qr_code_labels = ["2", "3", "4"],
         qr_code_target_axis = [None, "y", "-y"]):
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

    # find link ids by names
    links_to_visualize = ['trunk', 'leftThigh', 'rightThigh']
    num_joints = p.getNumJoints(exo_id, physicsClientId=p_client)
    link_ids = {}
    for i in range(num_joints):
        link_info = p.getJointInfo(exo_id, i, physicsClientId=p_client)
        link_name = link_info[12].decode('utf-8')
        if link_name in links_to_visualize:
            link_ids[link_name] = i

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
    leftThigh_QR2Virtual = np.array([
        [ 1, 0, 0, 0],
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
                continue
            else:
                if exo_frame_name == "leftThigh":
                    exo_frame_pose[exo_frame_name].append(qr_pose_data[exo_frame_name][i] @ leftThigh_QR2Virtual)
                elif exo_frame_name == "rightThigh":
                    exo_frame_pose[exo_frame_name].append(qr_pose_data[exo_frame_name][i] @ rightThigh_QR2Virtual)
                else:
                    exo_frame_pose[exo_frame_name].append(qr_pose_data[exo_frame_name][i])

    ###### Do inverse kinematics to find joint angles at every step ######
    for i in range(total_step):
        set_trunk_frame_pose(qr_pose_data["trunk"][i], exo_id, p_client)
        set_exo_pose(joint_angles_data[-1], exo_id, p_client)
        for exo_frame_name in exo_frame_names:
            if exo_frame_name == "trunk":
                continue
            else:
                joint_angles = get_joint_angles_and_set(exo_frame_pose[exo_frame_name][i], exo_id, link_ids[exo_frame_name], p_client)
        joint_angles_data.append(joint_angles)
    
    ###### Set the pose of exoskeleton in Pybullet simulation ######
    for i in range(total_step):
        set_trunk_frame_pose(qr_pose_data["trunk"][i], exo_id, p_client)
        set_exo_pose(joint_angles_data[i], exo_id, p_client)
        input("Press Enter to continue...")
    
    ###### Save the joint angles to npy file ######
    np.save("data/exo_pose.npy", joint_angles_data)            


    return 0

if __name__ == "__main__":
    main(qr_pose_path="data/0422_test/unified_poses.npy",
         exo_urdf_path="data/exo_model/urdf/exo_w_virtual_frame.urdf",
         exo_frame_names=["trunk", "leftThigh", "rightThigh"],
         qr_code_labels=["2", "3", "4"],
         qr_code_target_axis=[None, "y", "-y"])