import cv2
import numpy as np
import json
import os
import time
import pyboof as pb

pb.init_memmap()  # Initializes PyBoof

def save_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def calculate_qr_poses(image, qr_code_sizes, expected_labels, camera_matrix, dist_coeffs):
    """
    Calculates the poses of QR codes in an image.

    Args:
        image (numpy.ndarray): The input image.
        qr_code_sizes (list): List of sizes of the QR codes in meters.
        expected_labels (list): List of expected labels of the QR codes.
        camera_matrix (numpy.ndarray): The camera matrix.
        dist_coeffs (numpy.ndarray): The distortion coefficients.

    Returns:
        list: List of transformation matrices representing the poses of the QR codes.
    """

    detector = cv2.QRCodeDetector()
    retval, data, points, straight_qrcode = detector.detectAndDecodeMulti(image)
    transformations = {label: np.eye(4) for label in expected_labels}  # Default to identity matrix
    
    if points is not None:
        for i, label in enumerate(data):
            if label in expected_labels and points[i].shape[0] == 4:  # Ensure there are exactly 4 points
                idx = expected_labels.index(label)
                image_points = points[i]
                object_points = np.array([
                    [-qr_code_sizes[idx]/2, qr_code_sizes[idx]/2, 0],
                    [qr_code_sizes[idx]/2, qr_code_sizes[idx]/2, 0],
                    [qr_code_sizes[idx]/2, -qr_code_sizes[idx]/2, 0],
                    [-qr_code_sizes[idx]/2, -qr_code_sizes[idx]/2, 0]
                ], dtype=np.float32)

                retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
                if retval:
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = tvec[:, 0]
                    transformations[label] = T

    ordered_transformations = [transformations[label] for label in expected_labels]
    return ordered_transformations

def calculate_micro_qr_poses(image, qr_code_sizes, expected_labels, camera_matrix, dist_coeffs):
    """
    Calculates the poses of Micro QR codes in an image using PyBoof.

    Args:
        image (numpy.ndarray): The input image.
        qr_code_sizes (list): List of sizes of the QR codes in meters.
        expected_labels (list): List of expected labels of the QR codes.
        camera_matrix (numpy.ndarray): The camera matrix.
        dist_coeffs (numpy.ndarray): The distortion coefficients.

    Returns:
        list: List of transformation matrices representing the poses of the QR codes.
    """

    # Convert the input image to PyBoof format
    boof_image = pb.ndarray_to_boof(image)

    # Create a detector and detect Micro QR codes
    detector = pb.FactoryFiducial(np.uint8).microqr()
    detector.detect(boof_image)

    transformations = {label: np.eye(4) for label in expected_labels}  # Default to identity matrix

    for qr in detector.detections:
        label = qr.message
        if label in expected_labels:
            idx = expected_labels.index(label)
            
            image_points = np.array([[p.x, p.y] for p in qr.bounds.vertexes]).astype(np.float32)

            qr_size = qr_code_sizes[idx]
            object_points = np.array([
                [-qr_size/2, qr_size/2, 0],
                [qr_size/2, qr_size/2, 0],
                [qr_size/2, -qr_size/2, 0],
                [-qr_size/2, -qr_size/2, 0]
            ], dtype=np.float32)

            # SolvePnP to find the pose
            retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if retval:
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec[:, 0]
                transformations[label] = T

    ordered_transformations = [transformations[label] for label in expected_labels]
    return ordered_transformations

def get_qr_poses_from_video(video_path, qr_code_sizes, qr_labels, camera_matrix, dist_coeffs, process_freq=10, output_json_filename=None, save_to_file=False):
    """
    Extracts QR code poses from a video file.

    Args:
        video_path (str): Path to the video file.
        qr_code_sizes (list): List of QR code sizes in meters.
        qr_labels (list): List of QR code labels.
        camera_matrix (numpy.ndarray): Camera intrinsic matrix.
        dist_coeffs (numpy.ndarray): Camera distortion coefficients.
        process_freq (int, optional): Frequency of frames to process. Defaults to 10.
        output_json_filename (str, optional): Output JSON filename. Defaults to None.
        save_to_file (bool, optional): Whether to save the QR pose information to a file. Defaults to False.

    Returns:
        dict: Dictionary containing the QR pose information.

    Raises:
        IOError: If the video file cannot be opened.
    """
    if save_to_file and output_json_filename:
        output_json_filename = f"{output_json_filename}_{time.strftime('%Y%m%d%H%M%S')}.json"
        if not os.path.exists("data"):
            os.makedirs("data")
        output_json_path = os.path.join("data", output_json_filename)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video at path {video_path}.")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = max(int(video_fps / process_freq), 1)

    qr_pose_info = {label: [] for label in qr_labels} # qr_pose_info for current camera: qr_pose_info = {"1": [], "2": [], "3": [], "frame_number": 0, "occlusion_frame_number": {"1": 0, "2": 0, "3": 0}}
    qr_pose_info["frame_number"] = 0
    qr_pose_info["occlusion_frame_number"] = {label: 0 for label in qr_labels}
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_counter % skip_frames == 0:
            transformations = calculate_qr_poses(frame, qr_code_sizes, qr_labels, camera_matrix, dist_coeffs)
            for label, transformation in zip(qr_labels, transformations):
                if np.array_equal(transformation, np.identity(4)):
                    qr_pose_info["occlusion_frame_number"][label] += 1
                qr_pose_info[label].append(transformation.tolist())
            qr_pose_info["frame_number"] += 1
        
        frame_counter += 1

    cap.release()

    if qr_pose_info["frame_number"] > 0 and save_to_file:
        save_to_json(qr_pose_info, output_json_path)
        print(f"Saved QR pose information to {output_json_path}")
    elif not save_to_file:
        return qr_pose_info

# Example usage
if __name__ == "__main__":
    video_path = 'path/to/your/video.mp4'
    qr_code_sizes = [119, 119, 119]  # Sizes in millimeters
    qr_labels = ['1', '2', '3']
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])  # Placeholder values
    dist_coeffs = np.zeros((4, 1))  # Placeholder values

    # Assume save_to_file is True for demonstration
    get_qr_poses_from_video(video_path, qr_code_sizes, qr_labels, camera_matrix, dist_coeffs, save_to_file=True)
