import numpy as np
import json

from utils.camera_calibration import calibrate_camera_from_video
from utils.cal_poses_diff import cal_poses_diff
from utils.get_qr_pose import get_qr_poses_from_video
from utils.get_camera_pose import get_camera_pose_in_qr_frame

def main(calibration_video_path = None,
         calibration_data_path = None,
         calculation_video_path = None,
         camera_num = 2, 
         camera_names = ["camera1", "camera2"], 
         qr_labels = ["QR1", "QR2"], 
         qr_sizes = [0.1, 0.1], 
         base_qr_label = "QR1"):
    """
    Main function for camera calibration and QR code pose calculation.

    Args:
        camera_num (int): Number of cameras.
        camera_names (list): List of camera names.
        qr_labels (list): List of QR code labels.
        qr_sizes (list): List of QR code sizes in meters.
        base_qr_label (str): Label of the base QR code.

    Returns:
        None
    """
    try:
        if len(camera_names) != camera_num:
            raise ValueError("Number of camera names should match the number of cameras.")
    except ValueError as e:
        print(e)
        camera_names = [f"camera{i+1}" for i in range(camera_num)]
    
    ###### Initialization ######
    camera_dict = {}
    for i in range(camera_num):
        camera_dict[camera_names[i]] = {
            "Calibration_video_path": "",
            "Calibration_data_path": "",
            "Calibration_data": {
                "camera_matrix": [],
                "distortion_coefficients": []
            },
            "Calculation_video_path": "",
            "QR_pose_info": {},
            "Camera_pose": [],
        }

    ###### Camera calibration ######
    for i in range(camera_num):
        flag = input(f"Calibrate {camera_names[i]} with video? (y/n): ")
        if flag.lower() == 'y':
            if calibration_video_path[i] is not None:
                while True:
                    try:
                        path_to_video = input(f"Enter the path to the calibration video for \033[92m{camera_names[i]}\033[0m: ")
                        with open(path_to_video):
                            break
                    except FileNotFoundError:
                        print(f"File path \033[92m{path_to_video}\033[0m not found. Please enter a valid path.")

            camera_dict[camera_names[i]]["Calibration_video_path"] = path_to_video # record the path to the calibration video
            calibration_data = calibrate_camera_from_video(path_to_video, (9, 6), 0.025, 25, False)
            camera_dict[camera_names[i]]["Calibration_data"] = calibration_data # record the calibration data
        else:
            if calibration_data_path[i] is not None:
                while True:
                    try:
                        path_to_calib_data = input(f"Enter the path to the calibration JSON data for \033[92m{camera_names[i]}\033[0m: ")
                        with open(path_to_calib_data):
                            break
                    except FileNotFoundError:
                        print(f"File path \033[92m{path_to_calib_data}\033[0m not found. Please enter a valid path.")

            camera_dict[camera_names[i]]["Calibration_data_path"] = path_to_calib_data # record the path to the calibration data
            with open(path_to_calib_data, "r") as f:
                data = json.load(f)
                camera_dict[camera_names[i]]["Calibration_data"]["camera_matrix"] = np.array(data["camera_matrix"]) # record the calibration data
                camera_dict[camera_names[i]]["Calibration_data"]["distortion_coefficients"] = np.array(data["distortion_coefficients"]) # record the calibration data

    ###### Calculate QR code pose for each camera ######
    for i in range(camera_num):
        if calculation_video_path[i] is not None:
            while True:
                try:
                    path_to_video = input(f"Enter the path to the calculation video for \033[92m{camera_names[i]}\033[0m: ")
                    with open(path_to_video):
                        break
                except FileNotFoundError:
                    print(f"File path \033[92m{path_to_video}\033[0m not found. Please enter a valid path.")
    
        camera_dict[camera_names[i]]["Calculation_video_path"] = path_to_video # record the path to the video
        # calculate the QR code poses for each frame
        qr_pose_info = get_qr_poses_from_video(
            path_to_video,
            qr_sizes, # in meters
            qr_labels, # labels of QR code in sequence
            camera_dict[camera_names[i]]["Calibration_data"]["camera_matrix"], 
            camera_dict[camera_names[i]]["Calibration_data"]["distortion_coefficients"], 
            process_freq=10, 
            output_json_filename=None, 
            save_to_file=False)
        camera_dict[camera_names[i]]["QR_pose_info"] = qr_pose_info
        print(f"QR pose information for \033[92m{camera_names[i]}\033[0m obtained successfully.")
    
    ###### Construct camera positions with respect to the specified QR code ######
    if base_qr_label not in camera_dict[camera_names[i]]["QR_pose_info"]:
        raise ValueError(f"Wrong base_qr_label. Please provide a valid label.")
    for i in range(camera_num):
        qr_poses = camera_dict[camera_names[i]]["QR_pose_info"][base_qr_label] # transformation matrices for base QR code
        camera_pose = get_camera_pose_in_qr_frame([qr_poses]) # camera pose with respect to base QR code
        camera_dict[camera_names[i]]["Camera_pose"] = camera_pose # record the camera pose

    ###### Calculate QR code poses with respect to new camera poses ######
    for i in range(camera_num):
        qr_poses = camera_dict[camera_names[i]]["QR_pose_info"]
        for label in qr_labels:
            if label != base_qr_label: # for all QR codes except the base QR code
                qr_poses_new = []
                for qr_pose in qr_poses[label]:
                    if np.array_equal(qr_pose, np.identity(4)):
                        qr_pose_new = qr_pose
                    else:
                        qr_pose_new = np.dot(camera_dict[camera_names[i]]["Camera_pose"], qr_pose)
                    qr_poses_new.append(qr_pose_new.tolist())
                camera_dict[camera_names[i]]["QR_pose_info"][label] = qr_poses_new # update the QR pose information
    
    ###### Exclude the base QR code poses from the QR code poses ######
    for i in range(camera_num):
        camera_dict[camera_names[i]]["QR_pose_info"].pop(base_qr_label)
    qr_labels.remove(base_qr_label)
    qr_sizes.remove(qr_sizes[qr_labels.index(base_qr_label)])

    '''
    Now the list of the transformation matrices of QR code labeled "QR1" 
    with respect to the base QR code frame calculated from camera "camera1" is stored in 
    camera_dict["camera1"]["QR_pose_info"]["QR1"]
    '''

    ###### synchronize the QR code poses in different cameras ######

    # every QR code poses in same camera should have same length
    for i in range(camera_num):
        for j in range(1, len(qr_labels)):
            # compare the length of different qr_labels, raise error if there is a difference
            if len(camera_dict[camera_names[i]]["QR_pose_info"][qr_labels[j-1]]) != len(camera_dict[camera_names[i]]["QR_pose_info"][qr_labels[j]]):
                raise ValueError(f"QR code poses in camera {camera_names[i]} are not synchronized.")
    
    # synchronize the QR code poses in different cameras
    # find the camera with the most QR code poses
    max_len = 0
    max_len_camera = ""
    for i in range(camera_num):
        if len(camera_dict[camera_names[i]]["QR_pose_info"][qr_labels[0]]) > max_len:
            max_len = len(camera_dict[camera_names[i]]["QR_pose_info"][qr_labels[0]])
            max_len_camera = camera_names[i]
    basis_qr_poses = camera_dict[max_len_camera]["QR_pose_info"] # basis poses dictionary for QR codes
    basis_camera = max_len_camera

    # extend the basis_qr_poses at the beginning and the end by 10 identity matrices at each end
    for label in qr_labels:
        basis_qr_poses[label] = [np.identity(4)]*10 + basis_qr_poses[label] + [np.identity(4)]*10
    basis_len = len(basis_qr_poses[qr_labels[0]]) # length of basis poses

    # for each camera other than the basis camera, align the corresponding QR code poses with the basis poses, starting from the beginning
    for i in range(camera_num):
        if camera_names[i] != basis_camera:
            frame_len = len(camera_dict[camera_names[i]]["QR_pose_info"][qr_labels[0]]) # length of QR code poses for the current camera
            qr_poses = camera_dict[camera_names[i]]["QR_pose_info"]
            min_diff = float('inf')
            for starting_frame in range(0, basis_len-frame_len+1):
                basis_qr_poses_temp =\
                    {label: basis_qr_poses[label][starting_frame:starting_frame+frame_len] for label in qr_labels} # cut the basis poses
                diff = cal_poses_diff(qr_poses, basis_qr_poses_temp) # calculate the difference between the QR code poses for this part of basis poses
                print(f"Starting frame: {starting_frame}, Difference: {diff}")
                if diff < min_diff:
                    min_diff = diff
                    starting_frame_best = starting_frame # record the best starting frame
            print(f"Best starting frame for camera {camera_names[i]}: {starting_frame_best}")
            for label in qr_labels:
                camera_dict[camera_names[i]]["QR_pose_info"][label] =\
                    [np.identity(4)]*starting_frame_best +\
                    camera_dict[camera_names[i]]["QR_pose_info"][label] +\
                    [np.identity(4)]*(basis_len-starting_frame_best-frame_len) # align the QR code poses with the basis poses
    
    '''
    Now the list of the transformation matrices of QR code labeled "QR1" 
    with respect to the base QR code frame calculated from camera "camera1" is stored in 
    camera_dict["camera1"]["QR_pose_info"]["QR1"]
    All the QR code poses are synchronized across different cameras.
    i.e. camera_dict["camera1"]["QR_pose_info"]["QR1"][i] 
     and camera_dict["camera2"]["QR_pose_info"]["QR1"][i] 
     should be very close, i being the frame number, 
     except being identity matirx (QR code not detected in some cameras)
    ''' 
