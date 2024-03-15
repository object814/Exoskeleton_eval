import cv2
import numpy as np
import json
import os

def load_calibration_data(configs_path):
    # Load camera calibration data
    with open(configs_path, "r") as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["distortion_coefficients"])
    return camera_matrix, dist_coeffs

def calculate_qr_pose(image, qr_code_size, fx=None, fy=None, cx=None, cy=None, dist_coeffs=None, calibration_data_path=None):
    """
    Calculate the pose of a QR code in an image with respect to the camera.

    Args:
        image: The input image.
        qr_code_size: The size of the QR code in milimeters.
        fx, fy, cx, cy: The camera intrinsic parameters.
        dist_coeffs: The camera distortion coefficients.
        calibration_data_path: The path to the camera calibration data.
    Returns:
        The 4x4 transformation matrix of the QR code.
    """
    # Turn into meters
    qr_code_size /= 1000

    # Check if the camera intrinsic parameters are provided
    if fx is None or fy is None or cx is None or cy is None:
        # Load camera calibration data
        if calibration_data_path:
            camera_matrix, dist_coeffs = load_calibration_data(calibration_data_path)
        else:
            raise ValueError("Camera intrinsic parameters are not provided.")
    else:
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # Detect QR Code
    detector = cv2.QRCodeDetector()
    retval, points = detector.detect(image)
    
    if points is not None:
        # Points of QR code corners in the image
        image_points = points[0]
        
        # Points of QR code corners in real world
        object_points = np.array([
            [-qr_code_size/2, qr_code_size/2, 0],  # Top left
            [qr_code_size/2, qr_code_size/2, 0],   # Top right
            [qr_code_size/2, -qr_code_size/2, 0],  # Bottom right
            [-qr_code_size/2, -qr_code_size/2, 0]  # Bottom left
        ], dtype=np.float32)
        
        # Solve for pose
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Construct 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec[:, 0]
        
        return T
    else:
        print("No QR code detected.")
        return None

# Example usage:
# image = cv2.imread("path_to_your_image.jpg")
# transformation_matrix = calculate_qr_pose(image, qr_code_size=0.05, calibration_data_path="path_to_calibration_data.json")
# if transformation_matrix is not None:
#     print(transformation_matrix)
    
def process_video(video_path, qr_code_size, fx=None, fy=None, cx=None, cy=None, dist_coeffs=None, calibration_data_path=None, process_freq=10):
    """
    Process a video to calculate the pose of a QR code in each frame.

    Args:
        video_path: The path to the video file.
        qr_code_size: The size of the QR code in milimeters.
        fx, fy, cx, cy: The camera intrinsic parameters.
        dist_coeffs: The camera distortion coefficients.
        calibration_data_path: The path to the camera calibration data.
        process_freq: The desired processing frequency in Hz.
    """
    # Open video
    video_path = os.path.join("assets", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # Determine the number of frames to skip based on desired processing frequency
    skip_frames = int(video_fps / process_freq)
    
    if skip_frames == 0:
        skip_frames = 1  # Process every frame if video FPS is lower than the desired processing frequency

    frame_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        if frame_counter % skip_frames == 0:
            # Process the frame
            transformation_matrix = calculate_qr_pose(frame, qr_code_size, fx, fy, cx, cy, dist_coeffs, calibration_data_path)
            if transformation_matrix is not None:
                print(transformation_matrix)
        
        frame_counter += 1
    
    cap.release()
