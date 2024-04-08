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

def visualize_exo(urdf_path):
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    # Load exoskeleton URDF
    initialBasePosition = [0, 0, 1.5]
    initialBaseOrientation = p.getQuaternionFromEuler([0, 0, 0])
    exo_id = p.loadURDF(urdf_path, basePosition=initialBasePosition, baseOrientation=initialBaseOrientation, useFixedBase=True)

    # Correctly find the link IDs by names
    links_to_visualize = ['virtual_link_trunk', 'virtual_link_right', 'virtual_link_left']
    num_joints = p.getNumJoints(exo_id)
    link_ids = {}

    for i in range(num_joints):
        link_info = p.getJointInfo(exo_id, i)
        link_name = link_info[12].decode('utf-8')
        if link_name in links_to_visualize:
            link_ids[link_name] = i

    while True:
        p.stepSimulation()
        for link_name, link_id in link_ids.items():
            draw_frame(exo_id, link_id)
        time.sleep(0.01)  # Time step for simulation
    
    p.disconnect()

if __name__ == '__main__':
    visualize_exo('data/exo_model/urdf/exo_w_virtual_frame.urdf')
