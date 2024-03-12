import pybullet as p
import pybullet_data
import sys
import os
sys.path.append("../.")
from utils.init_camera import capture_image_and_depth

# Initialize the simulation
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load a plane
planeId = p.loadURDF("plane.urdf")

# Load a robot or any other object
# robotId = p.loadURDF("path_to_urdf_file.urdf", [0, 0, 1])

# Add a camera to the environment
capture_image_and_depth(p, visualization=True)

# Run the simulation for 1000 steps
for _ in range(1000):
    p.stepSimulation()
    
# Pause
input("Press Enter to continue...")

# Disconnect from the simulation
p.disconnect()