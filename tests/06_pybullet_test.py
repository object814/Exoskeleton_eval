import pybullet as p
import pybullet_data
import numpy as np
import time

# Load the transformation matrices
iphone_qr1_in_camera = np.load('data/qr1_in_iphone_0409.npy')
iphone_qr2_in_camera = np.load('data/qr2_in_iphone_0409.npy')
samsung_qr1_in_camera = np.load('data/qr1_in_samsung_0409.npy')
samsung_qr2_in_camera = np.load('data/qr2_in_samsung_0409.npy')

list1 = iphone_qr1_in_camera
list2 = iphone_qr2_in_camera
list3 = samsung_qr1_in_camera
list4 = samsung_qr2_in_camera

# Find the minimum length among all lists to ensure synchronization
min_length = min(len(list1), len(list2), len(list3), len(list4))
list1 = list1[:min_length]
list2 = list2[:min_length]
list3 = list3[:min_length]
list4 = list4[:min_length]

# Initialize PyBullet
p.connect(p.GUI)

def create_frame(label):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axis_ids = []
    for color in colors:
        id = p.addUserDebugLine([0, 0, 0], [0, 0, 0], lineColorRGB=color, lineWidth=4)
        axis_ids.append(id)
    text_id = p.addUserDebugText(text=label, textPosition=[0, 0, 0.1], textColorRGB=[1, 1, 1])
    return axis_ids, text_id

def update_frame(axis_ids, text_id, transformation_matrix, label):
    origin = transformation_matrix[:3, 3]
    for i, axis_id in enumerate(axis_ids):
        direction = transformation_matrix[:3, i]
        p.addUserDebugLine(origin, origin + direction, lineColorRGB=[i == j for j in range(3)], replaceItemUniqueId=axis_id, lineWidth=4)
    p.resetBasePositionAndOrientation(text_id, origin + [0, 0, 0.1], [0, 0, 0, 1])

def calculate_relative_transformation(T1, T2):
    T1_inv = np.linalg.inv(T1)
    T_rel = np.dot(T1_inv, T2)
    return T_rel

def is_identity_matrix(matrix):
    return np.allclose(matrix, np.eye(4), atol=1e-8)

# Create frames for all lists
origin_frame_1, origin_label_1 = create_frame("List 1 (Origin)")
relative_frame_1, relative_label_1 = create_frame("List 2 (Relative)")
origin_frame_2, origin_label_2 = create_frame("List 3 (Origin)")
relative_frame_2, relative_label_2 = create_frame("List 4 (Relative)")

# Visualization loop
index = 0
while index < min_length:
    if is_identity_matrix(list1[index]) or is_identity_matrix(list2[index]):
        update_frame(origin_frame_1, origin_label_1, np.eye(4), "List 1 (Origin)")
        update_frame(relative_frame_1, relative_label_1, np.eye(4), "List 2 (Relative)")
    else:
        update_frame(origin_frame_1, origin_label_1, np.eye(4), "List 1 (Origin)")
        T_rel_1 = calculate_relative_transformation(list1[index], list2[index])
        update_frame(relative_frame_1, relative_label_1, T_rel_1, "List 2 (Relative)")

    if is_identity_matrix(list3[index]) or is_identity_matrix(list4[index]):
        update_frame(origin_frame_2, origin_label_2, np.eye(4), "List 3 (Origin)")
        update_frame(relative_frame_2, relative_label_2, np.eye(4), "List 4 (Relative)")
    else:
        update_frame(origin_frame_2, origin_label_2, np.eye(4), "List 3 (Origin)")
        T_rel_2 = calculate_relative_transformation(list3[index], list4[index])
        update_frame(relative_frame_2, relative_label_2, T_rel_2, "List 4 (Relative)")

    index += 1
    input("Press Enter to proceed to the next transformation...")
    time.sleep(0.01)

# Disconnect PyBullet
p.disconnect()
