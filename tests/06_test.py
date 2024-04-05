import pybullet as p
import pybullet_data
import time

def main():
    # Start PyBullet in GUI mode
    physicsClient = p.connect(p.GUI)

    # Load the URDF of the plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    # Set the path to your exoskeleton's URDF file
    exoskeletonURDF = "data/exo_model/urdf/Complete_assemblt_electronics_3.urdf"

    # Define the initial base position and orientation
    initialBasePosition = [0, 0, 1]  # Example position: x=0, y=0, z=1 meters above the ground
    initialBaseOrientation = p.getQuaternionFromEuler([0, 0, 0])  # Example orientation: aligned with world frame

    # Load the URDF of the exoskeleton with the initial position and orientation
    exoskeletonId = p.loadURDF(exoskeletonURDF, basePosition=initialBasePosition, baseOrientation=initialBaseOrientation, useFixedBase=True)

    # Set gravity
    p.setGravity(0, 0, -10)

    # Wait for the user to press Enter
    input("Press Enter to change the pose of the exoskeleton...")

    # Define the new base position and orientation
    newBasePosition = [1, 1, 1]  # New position: x=1, y=1, z=1
    newBaseOrientation = p.getQuaternionFromEuler([0, 0, 45])  # Rotate 45 degrees around the Z-axis

    # Change the pose of the exoskeleton
    p.resetBasePositionAndOrientation(exoskeletonId, newBasePosition, newBaseOrientation)

    # Keep the simulation running. Exit with closing the PyBullet window or stopping the script.
    print("Pose changed. Close the PyBullet window or stop the script to exit.")
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    main()
