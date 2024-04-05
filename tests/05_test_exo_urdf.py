import pybullet as p
import pybullet_data
import numpy as np
import time

def visualize_exo(urdf_path):
    ''' Can drag the exoskeleton around to see the joints are performing correctly '''

    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load the plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    # Load exoskeleton urdf
    initialBasePosition = [0, 0, 1.5]
    initialBaseOrientation = p.getQuaternionFromEuler([0, 0, 0])

    exo_id = p.loadURDF(urdf_path, basePosition=initialBasePosition, baseOrientation=initialBaseOrientation, useFixedBase=True)

    while True:
        p.stepSimulation()
        time.sleep(0.01)  # Time step for simulation
    
    p.disconnect()
    

if __name__ == '__main__':
    visualize_exo('data/exo_model/urdf/Complete_assemblt_electronics_3.urdf')