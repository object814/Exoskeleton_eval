import cv2
import numpy as np
import os
import json
import argparse

def visualize_chessboard_detection(frame, corners):
    """
    Visualizes the chessboard detection by drawing corners on the frame.
    """
    cv2.drawChessboardCorners(frame, (args.n, args.m), corners, True)
    cv2.imshow('Chessboard Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calibrate_camera_from_video(video_filename, chessboard_dims, square_size, frames_to_use, save_to_file=True):
    """
    Calibrates the camera using a video with a chessboard pattern.
    
    Args:
        video_filename: Name of the video file located in the assets folder.
        chessboard_dims: Tuple of two integers with the number of inner corners in the chessboard pattern along the width and height respectively.
        square_size: Size of a square in the chessboard (in the same units used for the calibration object's real world dimensions).
        frames_to_use: Number of frames used for calibration.
        save_to_file: Boolean flag to specify whether to save the calibration data to a JSON file. Default is True.

    Returns:
        If save_to_file is False, returns a dictionary containing the camera matrix and distortion coefficients.
        If save_to_file is True, saves the calibration data to a JSON file located in the configs folder and returns None.
    """
    assets_path = os.path.join("assets", video_filename)
    configs_path = os.path.join("configs", "camera_calibration.json")
    
    objp = np.zeros((chessboard_dims[0]*chessboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_dims[0], 0:chessboard_dims[1]].T.reshape(-1, 2) * square_size
    
    objpoints = []
    imgpoints = []
    
    cap = cv2.VideoCapture(assets_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
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
                visualize_chessboard_detection(frame, corners)
                first_frame_visualized = True
            
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

    cap.release()

    if len(objpoints) > 0 and len(imgpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist)
        
        calibration_data = {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist()
        }
        
        if save_to_file:
            with open(configs_path, "w") as f:
                json.dump(calibration_data, f)
        
            print(f"Calibration data saved to {configs_path}")
        else:
            return calibration_data
    else:
        print("Could not find enough chessboard patterns for calibration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration from a video using a chessboard pattern.")
    parser.add_argument("video_filename", help="Name of the video file located in the assets folder")
    parser.add_argument("n", type=int, help="Number of inner corners in the chessboard pattern along the width")
    parser.add_argument("m", type=int, help="Number of inner corners in the chessboard pattern along the height")
    parser.add_argument("square_size", type=float, help="Size of a square in the chessboard (in the same units used for the calibration object's real world dimensions")
    parser.add_argument("--frame_to_use", type=int, help="Number of frames used for calibration", default=25)    
    args = parser.parse_args()
    
    calibrate_camera_from_video(args.video_filename, (args.n, args.m), args.square_size, args.frame_to_use)
