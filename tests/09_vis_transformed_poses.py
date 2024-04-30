import pybullet as p
import pybullet_data
import numpy as np
import time

# Load the transformation matrices
list1 = np.load('data/0409_test/qr2_final_poses.npy', allow_pickle=True).item()
list1 = list1['2']
# list2 = np.load('data/0409_test/samsung_qr2_in_qr1.npy')

# Cut the lists to the same length
# min_length = min(len(list1), len(list2))
list1 = list1[200:330]
# list2 = list2[200:330]

# Initialize PyBullet
p.connect(p.GUI)

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

# Create frames for all lists
frame_1 = create_frame()
frame_2 = create_frame()
frame_3 = create_frame()

# Visualization loop
index = 0
input("wait")
while index < len(list1):
    update_frame(frame_1, list1[index])
    # update_frame(frame_2, list2[index])

    index += 1
    print("current index: ", index)
    # input("Press Enter to proceed to the next transformation...")
    time.sleep(0.05)

# Disconnect PyBullet
p.disconnect()
