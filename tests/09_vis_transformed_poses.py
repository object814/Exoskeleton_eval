import pybullet as p
import pybullet_data
import numpy as np
import time

# Load the transformation matrices
poses = np.load('data/0416_test/final_poses.npy', allow_pickle='True').item()

list1 = poses['2']
list2 = poses['3']
list3 = poses['4']

print(len(list1))
print(len(list2))
print(len(list3))

# Initialize PyBullet
p.connect(p.GUI)

def create_frame():
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axis_ids = []
    for color in colors:
        id = p.addUserDebugLine([0, 0, 0], [0, 0, 0], lineColorRGB=color, lineWidth=4)
        axis_ids.append(id)
    return axis_ids

def update_frame(axis_ids, transformation_matrix, width=1):
    origin = transformation_matrix[:3, 3]
    for i, axis_id in enumerate(axis_ids):
        direction = transformation_matrix[:3, i]
        p.addUserDebugLine(origin, origin + direction, lineColorRGB=[i == j for j in range(3)], replaceItemUniqueId=axis_id, lineWidth=width)

# Create frames for all lists
frame_1 = create_frame()
frame_2 = create_frame()
frame_3 = create_frame()

# Visualization loop
index = 0
while index < len(list1):
    update_frame(frame_1, list1[index])
    update_frame(frame_2, list2[index])
    update_frame(frame_3, list3[index])

    index += 1
    input("Press Enter to proceed to the next transformation...")
    time.sleep(0.01)

# Disconnect PyBullet
p.disconnect()
