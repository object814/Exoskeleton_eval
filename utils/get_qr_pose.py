import cv2
import numpy as np
import json
import os
import time

def save_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def calculate_qr_poses(image, qr_code_sizes, expected_labels, camera_matrix, dist_coeffs):
    """
    Calculates the poses of QR codes in an image.

    Args:
        image (numpy.ndarray): The input image.
        qr_code_sizes (list): The sizes of the QR codes in millimeters.
        expected_labels (list): The expected labels of the QR codes.
        camera_matrix (numpy.ndarray): The camera matrix.
        dist_coeffs (numpy.ndarray): The distortion coefficients.

    Returns:
        list: A list of transformation matrices representing the poses of the QR codes.
    """
    qr_code_sizes = [size / 1000 for size in qr_code_sizes]
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecodeMulti(image)
    transformations = {label: np.eye(4) for label in expected_labels}
    
    if points is not None:
        for i, label in enumerate(data):
            if label in expected_labels:
                idx = expected_labels.index(label)
                image_points = points[i]
                object_points = np.array([
                    [-qr_code_sizes[idx]/2, qr_code_sizes[idx]/2, 0],
                    [qr_code_sizes[idx]/2, qr_code_sizes[idx]/2, 0],
                    [qr_code_sizes[idx]/2, -qr_code_sizes[idx]/2, 0],
                    [-qr_code_sizes[idx]/2, -qr_code_sizes[idx]/2, 0]
                ], dtype=np.float32)

                retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
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
        video_path (str): The path to the video file.
        qr_code_sizes (list): A list of QR code sizes in milimeters.
        qr_labels (list): A list of QR code labels.
        camera_matrix (numpy.ndarray): The camera matrix.
        dist_coeffs (numpy.ndarray): The distortion coefficients.
        process_freq (int, optional): The frequency at which to process frames. Defaults to 10.
        output_json_filename (str, optional): The name of the output JSON file. Defaults to None.
        save_to_file (bool, optional): Whether to save the QR pose information to a file. Defaults to False.

    Returns:
        dict: A dictionary containing the QR pose information. 
        The dictionary has the following structure:
        {
            "frame_number": int,  # The total number of processed frames.
            "label_1": [# List of 4x4 transformation matrices (as lists) for QR code with label "label_1"],
            "label_2": [...],  # Similarly for QR code with label "label_2".
            ...  # More labels as specified by `qr_labels` argument.
        }


    Raises:
        IOError: If the video file cannot be opened.
    """

    if save_to_file:
        output_json_filename = f"{output_json_filename or 'transformation_matrices'}_{time.strftime('%Y%m%d%H%M%S')}.json"
        output_json_path = os.path.join("data", output_json_filename)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video at path \033[92m{video_path}\033[0m.")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = max(int(video_fps / process_freq), 1)

    qr_pose_info = {label: [] for label in qr_labels}
    qr_pose_info["frame_number"] = 0
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_counter % skip_frames == 0:
            transformations = calculate_qr_poses(frame, qr_code_sizes, qr_labels, camera_matrix, dist_coeffs)
            for label, transformation in zip(qr_labels, transformations):
                qr_pose_info[label].append(transformation.tolist())
            qr_pose_info["frame_number"] += 1
        
        frame_counter += 1

    cap.release()

    if qr_pose_info["frame_number"] > 0:
        if save_to_file:
            save_to_json(qr_pose_info, output_json_path)
            print(f"Saved QR pose information to {output_json_path}")
        else:
            return qr_pose_info

if __name__ == "__main__":
    video_path = 'path/to/your/video.mp4'
    qr_code_sizes = [119, 119, 119] # Assuming same size for simplicity, adjust as needed
    qr_labels = ['1', '2', '3']
    calibration_data_path = 'configs/camera_calibration.json'
    output_json_filename = 'transformation_matrices'
    # Example call that saves to file
    get_qr_poses_from_video(video_path, qr_code_sizes, qr_labels, 10, calibration_data_path, output_json_filename, save_to_file=True)
    # Example call that returns the dictionary
    result = get_qr_poses_from_video(video_path, qr_code_sizes, qr_labels, 10, calibration_data_path, output_json_filename, save_to_file=False)
    # If you wish to print or further process `result`, you can do so here.

