import pybullet as p
import numpy as np
import cv2
import open3d as o3d

def capture_image_and_depth(pb_client, camera_pos=[0.5, 0.5, 1], target_pos=[0, 0, 0], intrinsic_params=[525, 525, 320, 240], visualization=False, depth=False):
    """
    Capture RGB image and depth information from the camera attached to the PyBullet environment
    Args:
    - pb_client: PyBullet client
    - camera_pos: list, position of the camera
    - target_pos: list, position of the target
    - intrinsic_params: list, intrinsic parameters of the camera
    - visualization: bool, whether to visualize the image
    - depth: bool, whether to capture depth information
    Returns:
    - rgb_img: RGB image in OpenCV format
    - depth_o3d: Depth information in Open3D format
    """
    # Set camera parameters
    view_matrix = pb_client.computeViewMatrix(camera_pos, target_pos, [0,0,1])
    projection_matrix = intrinsic_params
    
    # Capture RGB image
    width, height, rgb_img, _, _ = pb_client.getCameraImage(width=640, height=480, viewMatrix=view_matrix,
                                                     projectionMatrix=projection_matrix,
                                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)
    # Reshape image
    rgb_img = np.reshape(rgb_img, (height, width, 4))
    rgb_img = rgb_img[:, :, :3]  # Remove alpha channel

    # Convert to OpenCV format
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # Visualization
    if visualization:
        cv2.imshow("RGB Image", rgb_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Depth information
    if depth:
        depth_img = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                depth_img[y, x] = p.getClosestPoints([x, y], [x, y + 1], 0.01, -1, 0, 1)[0][0][8]
        # Convert depth image to Open3D format
        depth_o3d = o3d.geometry.Image(depth_img)
        return rgb_img, depth_o3d

    return rgb_img