import cv2
import numpy as np
import json
import os
import time

def save_transformation_matrices_to_json(matrices, file_path):
    """
    Save the transformation matrices to a JSON file.

    Args:
        matrices: The list of 4x4 transformation matrices.
        file_path: The path to save the JSON file.
    """
    # Convert numpy arrays to lists for JSON serialization
    matrices_list = [matrix.tolist() for matrix in matrices]
    
    with open(file_path, 'w') as f:
        json.dump(matrices_list, f)

def load_calibration_data(configs_path):
    # Load camera calibration data
    with open(configs_path, "r") as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["distortion_coefficients"])
    return camera_matrix, dist_coeffs

def calculate_qr_poses(image, qr_code_size, expected_labels, fx=None, fy=None, cx=None, cy=None, dist_coeffs=None, calibration_data_path=None):
    """
    Calculate the poses of multiple QR codes in an image with respect to the camera.

    Args:
        image: The input image.
        qr_code_size: The size of the QR codes in millimeters.
        expected_labels: A list of expected QR code labels in order.
        fx, fy, cx, cy: The camera intrinsic parameters.
        dist_coeffs: The camera distortion coefficients.
        calibration_data_path: The path to the camera calibration data.
    Returns:
        A list of 4x4 transformation matrices for the QR codes, ordered by expected_labels.
        Identity matrices are returned for undetected QR codes.
    """
    # Convert QR code size to meters
    qr_code_size /= 1000

    # Initialize camera matrix
    if fx is None or fy is None or cx is None or cy is None:
        if calibration_data_path:
            camera_matrix, dist_coeffs = load_calibration_data(calibration_data_path)
        else:
            raise ValueError("Camera intrinsic parameters are not provided.")
    else:
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Detect and decode all QR codes in the image
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecodeMulti(image)
    
    # Create a dictionary to store transformation matrices for detected QR codes
    transformations = {label: np.eye(4) for label in expected_labels}
    
    if points is not None:
        for i, label in enumerate(data):
            if label in expected_labels:
                image_points = points[i]
                object_points = np.array([
                    [-qr_code_size/2, qr_code_size/2, 0],  # Top left
                    [qr_code_size/2, qr_code_size/2, 0],   # Top right
                    [qr_code_size/2, -qr_code_size/2, 0],  # Bottom right
                    [-qr_code_size/2, -qr_code_size/2, 0]  # Bottom left
                ], dtype=np.float32)

                # Solve for pose
                retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

                # Convert rotation vector to rotation matrix and create the transformation matrix
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec[:, 0]

                transformations[label] = T

    # Order transformation matrices according to expected_labels
    ordered_transformations = [transformations[label] for label in expected_labels]

    return ordered_transformations

# Example usage:
# image = cv2.imread("path_to_your_image.jpg")
# transformation_matrix = calculate_qr_pose(image, qr_code_size=0.05, calibration_data_path="path_to_calibration_data.json")
# if transformation_matrix is not None:
#     print(transformation_matrix)
    
def process_video(video_path, qr_code_size, labels, process_freq=10, calibration_data_path=None, output_json_filename=None):
    """
    Process a video and save the transformation matrices of the QR codes to a JSON file.

    Args:
        video_path: The path to the input video.
        qr_code_size: The size of the QR code in milimeters.
        labels: A list of expected QR code labels in order. e.g. ['1', '2', '3']
        process_freq: The frequency of frames to process.
        calibration_data_path: The path to the camera calibration data.
        output_json_path: The path to save the transformation matrices.
    """
    # Read in the filename
    if output_json_filename is None:
        output_json_filename = f"transformation_matrices_{time.strftime('%Y%m%d%H%M%S')}.json"
    elif output_json_filename.endswith(".json"):
        output_json_filename = output_json_filename
    else:
        output_json_filename = f"{output_json_filename}.json"

    # Check if the output file already exists
    output_json_path = os.path.join("data", output_json_filename)
    if os.path.isfile(output_json_path):
        print(f"Warning: The file {output_json_path} already exists and will be overwritten.")

    # Open the video
    video_path = os.path.join("assets", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get the video frame rate and calculate the number of frames to skip
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = max(int(video_fps / process_freq), 1)
    
    frame_counter = 0
    
    # Collect the transformation matrices
    transformation_matrix_dic_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        if frame_counter % skip_frames == 0:
            # Process the frame
            transformation_matrix_dic = calculate_qr_poses(frame, qr_code_size, labels, calibration_data_path=calibration_data_path)
            transformation_matrix_dic_list.append(transformation_matrix_dic)
        
        frame_counter += 1
    
    cap.release()
    
    # Save the collected transformation matrices to a JSON file
    if transformation_matrix_dic_list:
        save_transformation_matrices_to_json(transformation_matrix_dic_list, output_json_path)
        print(f"Saved transformation matrices to {output_json_path}")
        return transformation_matrix_dic_list
    else:
        print("No transformation matrices to save.")
        return None

if __name__ == "__main__":
    process_video('samsung_s23_test.mp4', 119, 10, 'configs/camera_calibration.json', 'transformation_matrices.json')