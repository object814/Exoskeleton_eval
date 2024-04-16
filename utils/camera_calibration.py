import cv2
import numpy as np
import os
import sys
import json
import argparse

def visualize_chessboard_detection(frame, chessboard_dims, corners):
    """
    Visualizes the chessboard detection by drawing corners on the frame.
    """
    cv2.drawChessboardCorners(frame, chessboard_dims, corners, True)
    cv2.imshow('Chessboard Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calibrate_camera_from_video(assets_path, save_path, chessboard_dims, square_size, frames_to_use, save_to_file=True):
    """
    Calibrates the camera using a video file containing chessboard patterns.

    Args:
        assets_path (str): The path to the video file.
        save_path (str): The path to save the calibration data.
        chessboard_dims (tuple): The dimensions of the chessboard pattern (rows, columns).
        square_size (float): The size of each square in the chessboard pattern.
        frames_to_use (int): The number of frames to use for calibration.
        save_to_file (bool, optional): Whether to save the calibration data to a file. Defaults to True.

    Returns:
        dict or None: If `save_to_file` is False, returns a dictionary containing the calibration data.
                     If `save_to_file` is True, returns None.

    Raises:
        FileNotFoundError: If the video file is not found.
        ValueError: If the number of frames to use is less than or equal to 0.

    """
    configs_path = save_path
    
    objp = np.zeros((chessboard_dims[0]*chessboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_dims[0], 0:chessboard_dims[1]].T.reshape(-1, 2) * square_size
    
    objpoints = []
    imgpoints = []
    
    cap = cv2.VideoCapture(assets_path)
    if not cap.isOpened():
        raise FileNotFoundError("Error: Could not open video.")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (chessboard_dims[0], chessboard_dims[1]), None)
        
        if ret:
            frame_indices.append(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)  # Append current frame index
    
    cap.release()
    
    # If less than 25 frames have the chessboard, use all of them
    if len(frame_indices) < frames_to_use:
        selected_indices = frame_indices
    else:
        step = len(frame_indices) / frames_to_use
        selected_indices = [frame_indices[int(i * step)] for i in range(frames_to_use)]

    # Re-open the video and process selected frames
    cap = cv2.VideoCapture(assets_path)
    first_frame_visualized = False
    for index in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (chessboard_dims[0], chessboard_dims[1]), None)
        
        if ret:
            if not first_frame_visualized:
                visualize_chessboard_detection(frame, chessboard_dims, corners)
                first_frame_visualized = True
            
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

    cap.release()

    if len(objpoints) > 0 and len(imgpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist)
        
        if save_to_file:
            calibration_data = {
                "camera_matrix": mtx.tolist(),
                "distortion_coefficients": dist.tolist()
            }
            with open(configs_path, "w") as f:
                json.dump(calibration_data, f)
        
            print(f"Calibration data saved to {configs_path}")
        else:
            calibration_data = {
                "camera_matrix": np.array(mtx),
                "distortion_coefficients": np.array(dist)
            }
            return calibration_data
    else:
        print("Could not find enough chessboard patterns for calibration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration from a video using a chessboard pattern.")
    parser.add_argument("video_filename", help="Path to the video file")
    parser.add_argument("save_config_filename", help="Path to the config file")
    parser.add_argument("n", type=int, help="Number of inner corners in the chessboard pattern along the width")
    parser.add_argument("m", type=int, help="Number of inner corners in the chessboard pattern along the height")
    parser.add_argument("square_size", type=float, help="Size of a square in the chessboard (in the same units used for the calibration object's real world dimensions")
    parser.add_argument("--frame_to_use", type=int, help="Number of frames used for calibration", default=25)    
    parser.add_argument("--save_to_file", type=int, help="Save the result as file", default=True)    
    args = parser.parse_args()
    
    calibrate_camera_from_video(args.video_filename, args.save_config_filename, (args.n, args.m), args.square_size, args.frame_to_use)
