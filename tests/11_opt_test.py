import pybullet as p
import numpy as np
from scipy.optimize import minimize

def get_current_axis_dir(exo_id, link_id, axis, physicsClientId):
    # Draws XYZ frame for a given object and link ID.
    link_state = p.getLinkState(exo_id, link_id, computeForwardKinematics=True, physicsClientId=physicsClientId)
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

def cost_function(joint_angles, target_opt_dir, opt_link_name, opt_axis, opt_joints, link_index, exo_id, client_id):
    # Set joint angles
    for i, angle in enumerate(joint_angles):
        p.resetJointState(exo_id, opt_joints[i], angle, physicsClientId=client_id)
        # p.setJointMotorControl2(exo_id, opt_joints[i], p.POSITION_CONTROL, targetPosition=angle, physicsClientId=client_id)
    # p.stepSimulation(physicsClientId=client_id)

    # Get the direction of the optimization link
    opt_dir = get_current_axis_dir(exo_id, link_index[opt_link_name], opt_axis, client_id)

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

def optimize_joints(target_opt_dir, opt_link_name, opt_axis, opt_joints, link_index, opt_exo_id, opt_client, iterations=100, learning_rate=0.01, threshold=0.01):
    # Normalize the target direction vector
    target_opt_dir = np.array(target_opt_dir) / np.linalg.norm(target_opt_dir)
    # Random initial joint angles
    # initial_opt_joint_angles = np.random.uniform(-np.pi, np.pi, len(opt_joints))
    # for i in range(len(opt_joints)):
    #     p.resetJointState(opt_exo_id, opt_joints[i], initial_opt_joint_angles[i], physicsClientId=opt_client)
    # Hyperparameters
    learning_rate = learning_rate
    threshold = threshold

    for _ in range(iterations):
        gradients = np.zeros(len(opt_joints)) # Initialize gradients
        current_opt_joint_angles = []
        for joint_id in opt_joints:
            current_opt_joint_angles.append(p.getJointState(opt_exo_id, joint_id, physicsClientId=opt_client)[0])

        for i in range(len(opt_joints)):
            perturbed_opt_joint_angles = current_opt_joint_angles.copy()
            perturbed_opt_joint_angles[i] += 0.01
            perturbed_cost = cost_function(perturbed_opt_joint_angles, 
                                           target_opt_dir, 
                                           opt_link_name, 
                                           opt_axis, 
                                           opt_joints, 
                                           link_index, 
                                           opt_exo_id, 
                                           opt_client)
            
            current_cost = cost_function(current_opt_joint_angles,
                                         target_opt_dir,
                                         opt_link_name,
                                         opt_axis,
                                         opt_joints,
                                         link_index,
                                         opt_exo_id,
                                         opt_client)

            gradient = (perturbed_cost - current_cost) / 0.01
            gradients[i] = gradient

        # Update joint angles
        new_opt_joint_angles = current_opt_joint_angles - learning_rate * gradients
        # for i in range(len(opt_joints)):
        #     p.resetJointState(opt_exo_id, opt_joints[i], new_opt_joint_angles[i], physicsClientId=opt_client)
        # Calculate cost
        cost = cost_function(new_opt_joint_angles, target_opt_dir, opt_link_name, opt_axis, opt_joints, link_index, opt_exo_id, opt_client)
        if cost < threshold:
            print(f"Optimization converged at iteration {_}")
            break

    # Calculate final error
    final_opt_joint_angles = []
    for joint_id in opt_joints:
            final_opt_joint_angles.append(p.getJointState(opt_exo_id, joint_id, physicsClientId=opt_client)[0])
    final_cost = cost_function(final_opt_joint_angles, target_opt_dir, opt_link_name, opt_axis, opt_joints, link_index, opt_exo_id, opt_client)
    final_dir = get_current_axis_dir(opt_exo_id, link_index[opt_link_name], opt_axis, opt_client)
    angle_radians = np.arccos(np.clip(np.dot(final_dir, target_opt_dir) / (np.linalg.norm(final_dir) * np.linalg.norm(target_opt_dir)), -1.0, 1.0))
    angle_degrees = angle_radians / np.pi * 180

    print(f"Final cost: {final_cost}")
    print(f"Final direction: {final_dir}")
    print(f"Final difference: {angle_degrees} degrees")

    return final_opt_joint_angles

# Pybullet initialization
vis_client = p.connect(p.GUI)
p.setGravity(0, 0, -9.81, physicsClientId=vis_client)
p.setTimeStep(1/240, physicsClientId=vis_client)

opt_client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81, physicsClientId=opt_client)
p.setTimeStep(1/240, physicsClientId=opt_client)

# Load URDF model
vis_exo_id = p.loadURDF('data/exo_model/urdf/exo_w_virtual_frame.urdf', useFixedBase=True, physicsClientId=vis_client)
opt_exo_id = p.loadURDF('data/exo_model/urdf/exo_w_virtual_frame.urdf', useFixedBase=True, physicsClientId=opt_client)

# Declare links to be optimized
link_names = ['trunk', 'rightThigh', 'leftThigh']

# Get link indeces
link_index = {}
for link_name in link_names:
    for i in range(p.getNumJoints(vis_exo_id, physicsClientId=vis_client)):
        if p.getJointInfo(vis_exo_id, i, vis_client)[12].decode('utf-8') == link_name:
            link_index[link_name] = i
            break
print(link_index)

# Record initial joint positions
initial_positions = [p.getJointState(vis_exo_id, i, physicsClientId=vis_client)[0] for i in range(p.getNumJoints(vis_exo_id, physicsClientId=vis_client))]

# Declare optimization link
opt_link_name = input("Enter the link name to optimize: ")
if opt_link_name == 'r':
    opt_link_name = 'rightThigh'
    opt_axis = '-y'
elif opt_link_name == 'l':
    opt_link_name = 'leftThigh'
    opt_axis = 'y'

# Find the kinematic chain for the optimization link
kinematic_chain = get_kinematic_chain(vis_exo_id, link_index[opt_link_name], physicsClientId=vis_client)
opt_joints = kinematic_chain

initial_target_opt_dir = get_current_axis_dir(vis_exo_id, link_index[opt_link_name], opt_axis, physicsClientId=vis_client)
print(f'Initial target direction: {initial_target_opt_dir}')
input('PAUSE')

# Run simulation to let user move the robot to desired pose
while True:
    p.stepSimulation(physicsClientId=vis_client)
    # print once
    print('Move the robot to the desired pose and press "Enter" to start optimization.')
    # draw the axis
    draw_axis(vis_exo_id, link_index[opt_link_name], opt_axis, physicsClientId=vis_client)
    keys = p.getKeyboardEvents(physicsClientId=vis_client)
    # if user presses Enter
    if 65309 in keys:
        target_opt_dir = get_current_axis_dir(vis_exo_id, link_index[opt_link_name], opt_axis, physicsClientId=vis_client)
        # print opt_joint angles
        current_opt_joint_angles = []
        for joint_id in opt_joints:
            current_opt_joint_angles.append(p.getJointState(opt_exo_id, joint_id, physicsClientId=vis_client)[0])
        p.removeAllUserDebugItems(physicsClientId=vis_client)
        break

# Reset the robot to initial pose
for i, pos in enumerate(initial_positions):
    p.resetJointState(vis_exo_id, i, pos, physicsClientId=vis_client)
for i, pos in enumerate(initial_positions):
    p.resetJointState(opt_exo_id, i, pos, physicsClientId=opt_client)

print(target_opt_dir)
print(current_opt_joint_angles)
input('Press "Enter" to start optimization.')
# target_opt_dir = np.array([0.13966019,  0.99002722, -0.01846987])
# target_opt_dir = target_opt_dir / np.linalg.norm(target_opt_dir)
# ground_truth_opt_angles = np.array([0.44922844406596635, -0.19987191331233772])

''' Optimization with Scipy '''
# random initial positions
# print(f'Number of joints to optimize: {len(opt_joints)}')
# initial_positions = np.random.uniform(-np.pi, np.pi, len(opt_joints))

# BFGS
# result = minimize(
#         cost_function,
#         initial_positions,
#         args=(target_opt_dir, opt_link_name, opt_axis, opt_joints, link_index, opt_exo_id, opt_client),
#         method='BFGS',
#         options={'disp': True, 'gtol': 1e-2, 'maxiter': 1000}
#     )

# Powell
# result = minimize(cost_function,
#                   initial_positions,
#                   args=(target_opt_dir, opt_link_name, opt_axis, opt_joints, link_index, opt_exo_id, opt_client), 
#                   method='Powell',
#                   options={'xtol': 1e-8, 'ftol': 1e-8, 'disp': True})

# L-BFGS-B
# bounds = [(-np.pi, np.pi) for _ in initial_positions]

# result = minimize(cost_function,
#                   initial_positions,
#                   args=(target_opt_dir, opt_link_name, opt_axis, opt_joints, link_index, opt_exo_id, opt_client), 
#                   method='L-BFGS-B',
#                   bounds=bounds,
#                   options={'disp': True})

# Visualization
# set joint angles to the optimized values
# print(result.x)
# for i, angle in enumerate(result.x):
#         p.resetJointState(vis_exo_id, opt_joints[i], angle, physicsClientId=vis_client)

''' Optimization with custom gradient descent '''
result = optimize_joints(target_opt_dir, opt_link_name, opt_axis, opt_joints, link_index, opt_exo_id, opt_client)
for i, angle in enumerate(result):
        p.resetJointState(vis_exo_id, opt_joints[i], angle, physicsClientId=vis_client)


# draw the target direction
p.addUserDebugLine([0, 0, 0], target_opt_dir, [0, 1, 0], lineWidth=5, physicsClientId=vis_client) # green
# draw the current direction
current_dir = get_current_axis_dir(vis_exo_id, link_index[opt_link_name], opt_axis, physicsClientId=vis_client)
p.addUserDebugLine([0, 0, 0], current_dir, [1, 0, 0], lineWidth=5, physicsClientId=vis_client) # red

# Output error
input('Press "Enter" to exit.')
p.disconnect()