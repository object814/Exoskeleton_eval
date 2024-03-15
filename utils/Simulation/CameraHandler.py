import numpy as np
import open3d as o3d
import cv2
import math

class CameraHandler:
    def __init__(self, pb_client, camera_pos=[0, 0, 1], target_pos=[1, 0, 1], intrinsic_params=[525, 525, 320, 240], image_size=[640, 480]):
        """
        Initializes a CameraHandler object.

        Args:
            pb_client (object): The pb_client object used for communication with the physics server.
            camera_pos (list, optional): The position of the camera in the world coordinates. Defaults to [0, 0, 1].
            target_pos (list, optional): The position of the target that the camera is looking at in the world coordinates. Defaults to [1, 0, 1].
            intrinsic_params (list, optional): The intrinsic parameters of the camera. Defaults to [525, 525, 320, 240].
            image_size (list, optional): The size of the image captured by the camera. Defaults to [640, 480].
        """
        self.pb_client = pb_client
        self.camera_pos = camera_pos
        self.target_pos = target_pos
        self.fx, self.fy, self.cx, self.cy = intrinsic_params
        self.width, self.height = image_size
        self.aspect_ratio = self.width / self.height
        self.fov = self.calculate_fov(self.fx, self.width)
        self.view_matrix = self.pb_client.computeViewMatrix(camera_pos, target_pos, [0, 0, 1])
        self.projection_matrix = self.pb_client.computeProjectionMatrixFOV(fov=self.fov, aspect=self.aspect_ratio, nearVal=0.1, farVal=100.0)

    def calculate_fov(self, focal_length, image_width):
        """
        Calculate the field of view from the focal length and image width.

        Args:
            focal_length (float): The focal length in pixels.
            image_width (int): The width of the image in pixels.

        Returns:
            float: The field of view in degrees.
        """
        # Assuming the sensor width matches the image width in this approximation
        fov_rad = 2 * math.atan(image_width / (2 * focal_length))
        fov_deg = math.degrees(fov_rad)
        return fov_deg
    
    def capture_image_and_depth(self, visualization=False):
        """
        Capture the RGB image and the point cloud from the camera in the simulation.

        Args:
            visualization (bool, optional): A boolean indicating whether to visualize the RGB image and the point cloud. Defaults to False.

        Returns:
            numpy.ndarray: The RGB image in OpenCV format.
            o3d.geometry.PointCloud: The point cloud in Open3D format.
        """
        width, height = 640, 480
        img = self.pb_client.getCameraImage(width, height, viewMatrix=self.view_matrix,
                                            projectionMatrix=self.projection_matrix,
                                            renderer=self.pb_client.ER_BULLET_HARDWARE_OPENGL)
        rgb_img = np.array(img[2])
        depth_buffer = np.array(img[3])

        # Process depth buffer
        depth = self._process_depth_buffer(depth_buffer, width, height)

        # Convert to OpenCV format
        rgb_img_cv = cv2.cvtColor(rgb_img.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)

        # Generate point cloud from depth image
        point_cloud = self._generate_point_cloud(depth, width, height, rgb_img_cv)

        if visualization:
            cv2.imshow("RGB Image", rgb_img_cv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            o3d.visualization.draw_geometries([point_cloud])

        return rgb_img_cv, point_cloud

    def _process_depth_buffer(self, depth_buffer, width, height):
        """
        Process the depth buffer to obtain the depth values.

        Args:
            depth_buffer (numpy.ndarray): The depth buffer.
            width (int): The width of the depth buffer.
            height (int): The height of the depth buffer.

        Returns:
            numpy.ndarray: The processed depth values.
        """
        far = 100.0
        near = 0.1
        depth = far * near / (far - (far - near) * depth_buffer)
        return depth

    def _generate_point_cloud(self, depth_img, width, height, rgb_img):
        """
        Generates a point cloud from a depth image and an RGB image.

        Args:
            depth_img (numpy.ndarray): The depth image.
            width (int): The width of the images.
            height (int): The height of the images.
            rgb_img (numpy.ndarray): The RGB image.

        Returns:
            o3d.geometry.PointCloud: The generated point cloud.
        """
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # Correcting the depth image for coordinate system differences between OpenCV and Open3D
        depth_img_corrected = np.copy(depth_img)
        # depth_img_corrected = np.flipud(depth_img_corrected)  # Flip the depth image vertically

        depth_o3d = o3d.geometry.Image((depth_img_corrected * 1000).astype(np.uint16))
        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=100.0, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        return pcd
