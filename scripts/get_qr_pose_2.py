import numpy as np
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.camera_calibration import calibrate_camera_from_video
from utils.pose_utils import cal_poses_diff, is_similar, interpolate_transform, interpolate_transform, ave_pose
from utils.get_qr_pose import get_qr_poses_from_video
from utils.get_camera_pose import get_camera_pose_in_qr_frame

def consecutive_idx(index_list):
    # Initialize an empty list to hold the result
    grouped_indexes = []
    # Initialize a temporary list to hold consecutive indexes
    temp_list = []
    # Iterate through the list of indexes
    for i, index in enumerate(index_list):
        # Add the first index to the temporary list
        if i == 0:
            temp_list.append(index)
        else:
            # Check if the current index is consecutive to the previous index
            if index == index_list[i-1] + 1:
                temp_list.append(index)
            else:
                # If not consecutive, add the temp_list to the grouped_indexes and start a new temp_list
                grouped_indexes.append(temp_list)
                temp_list = [index]
    # Add the last temp_list to the grouped_indexes
    grouped_indexes.append(temp_list)
    return grouped_indexes

def main(calib_video_path = None,
         calib_data_path = None,
         calib_chessboard_size = None,
         calib_chessboard_square_size = None,
         cam_pose_calib_video_path = None,
         operational_video_path = None,
         camera_num = 2,
         frame_rate = 10,
         camera_names = ["camera1", "camera2"], 
         qr_labels = ["1", "2"], 
         qr_sizes = [0.1, 0.1], 
         base_qr_label = "QR1",
         test_name = "test",
         final_pose_path = None,
         debug = False):
    """
    Main function for camera calibration and QR code pose calculation

    Pipeline:
    1. Camera calibration with chessboard pattern: user can either provide the path to the calibration video or the path to the calibrated data for each camera
    2. Calculate QR code poses for each camera: user needs to provide the path to the calculation video for each camera
        The QR code poses here are with respect to the camera frame, frames with no QR code detected are identity matrices
    3. Construct camera positions with respect to the specified QR code: user needs to provide the label of the base QR code
        In real world, the base QR code should be a fixed QR code, e.g. the QR code on the floor
        It is not recommended to move the camera during the recording of the calculation video, since it will affect the overall accuracy
        Every camera should have the base QR code in its field of view
    4. Calculate QR code poses with respect to camera poses
        This turns same QR code in different cameras into the reference frame of the base QR code
        Frames with no QR code detected are identity matrices
    5. Synchronize the QR code poses in different cameras
        The best matching sequence of QR code poses in different cameras is found
        All cameras are cut to the same length
    6. Calculate the average pose for each frame
        For frames with more than one valid QR code pose, the average pose is calculated
        For frames with no valid QR code pose, linear interpolation is used to fill the gap
        For frames with only one valid QR code pose, the pose is kept
        Before calculating the average pose, unstable QR code poses are detected and replaced with identity matrices

    Args:
        calib_video_path (str, optional): Path to the calibration video for each camera. Defaults to None.
        calib_data_path (str, optional): Path to the calibrated data for each camera. Defaults to None.
        calib_chessboard_size (tuple, optional): Size of the chessboard pattern used for calibration. Defaults to None.
        calib_chessboard_square_size (float, optional): Size of each square in the chessboard pattern in meters. Defaults to None.
        operational_video_path (str, optional): Path to the calculation video for each camera. Defaults to None.
        camera_num (int, optional): Number of cameras. Defaults to 2.
        frame_rate (int, optional): Frame rate of the calculation video. Defaults to 10.
        camera_names (list, optional): List of camera names. Defaults to ["camera1", "camera2"].
        qr_labels (list, optional): List of QR code labels. Defaults to ["1", "2"].
        qr_sizes (list, optional): List of QR code sizes in meters. Defaults to [0.1, 0.1].
        base_qr_label (str, optional): Label of the base QR code. Defaults to "QR1".

    Returns:
        None
    """
    try:
        if len(camera_names) != camera_num:
            raise ValueError("Number of camera names should match the number of cameras.")
    except ValueError as e:
        print(e)
        camera_names = [f"camera{i+1}" for i in range(camera_num)]

    # Create folder to store data if not exists
    if not os.path.exists(f"data/{test_name}"):
        os.makedirs(f"data/{test_name}")

    ###### Initialization ######
    camera_dict = {}
    for i in range(camera_num):
        camera_dict[camera_names[i]] = {
            "calib_video_path": "",
            "calib_data_path": "",
            "Calibration_data": {
                "camera_matrix": [],
                "distortion_coefficients": []
            },
            "operational_video_path": "",
            "cam_pose_calib_video_path": "",
            "QR_pose_info": {},
            "Camera_pose": [],
        }

    ###### Camera calibration ######
    for i in range(camera_num):
        print("----------------------------------------------")
        flag = input(f"Calibrate {camera_names[i]} with video? (y/n): ")
        if flag.lower() == 'y':
            if calib_video_path[i] is None:
                while True:
                    try:
                        path_to_video = input(f"Enter the path to the calibration video for \033[92m{camera_names[i]}\033[0m: ")
                        with open(path_to_video):
                            break
                    except FileNotFoundError:
                        print(f"File path \033[92m{path_to_video}\033[0m not found. Please enter a valid path.")
            else:
                path_to_video = calib_video_path[i]
            
            print(path_to_video)
            camera_dict[camera_names[i]]["calib_video_path"] = path_to_video # record the path to the calibration video
            calibration_data = calibrate_camera_from_video(path_to_video, None, calib_chessboard_size, calib_chessboard_square_size, 30, False)
            camera_dict[camera_names[i]]["Calibration_data"] = calibration_data # record the calibration data
        else:
            if calib_data_path[i] is None:
                while True:
                    try:
                        path_to_calib_data = input(f"Enter the path to the calibration JSON data for \033[92m{camera_names[i]}\033[0m: ")
                        with open(path_to_calib_data):
                            break
                    except FileNotFoundError:
                        print(f"File path \033[92m{path_to_calib_data}\033[0m not found. Please enter a valid path.")
            else:
                path_to_calib_data = calib_data_path[i]

            camera_dict[camera_names[i]]["calib_data_path"] = path_to_calib_data # record the path to the calibration data
            print(f"Calibration data for \033[92m{camera_names[i]}\033[0m loaded successfully: {path_to_calib_data}")
            with open(path_to_calib_data, "r") as f:
                data = json.load(f)
                camera_dict[camera_names[i]]["Calibration_data"]["camera_matrix"] = np.array(data["camera_matrix"]) # record the calibration data
                camera_dict[camera_names[i]]["Calibration_data"]["distortion_coefficients"] = np.array(data["distortion_coefficients"]) # record the calibration data

    ###### Calculate QR code pose in operational video for each camera ######
    for i in range(camera_num):
        if operational_video_path[i] is None:
            while True:
                try:
                    path_to_video = input(f"Enter the path to the calculation video for \033[92m{camera_names[i]}\033[0m: ")
                    with open(path_to_video):
                        break
                except FileNotFoundError:
                    print(f"File path \033[92m{path_to_video}\033[0m not found. Please enter a valid path.")
        else:
            path_to_video = operational_video_path[i]
        
        camera_dict[camera_names[i]]["operational_video_path"] = path_to_video # record the path to the video
        print("----------------------------------------------")
        print(f"Calculating QR code poses for \033[92m{camera_names[i]}\033[0m, video: {path_to_video}")
        print("Starting calculation...")
        # calculate the QR code poses for each frame
        qr_pose_info = get_qr_poses_from_video(
            path_to_video,
            qr_sizes, # in meters
            qr_labels, # labels of QR code in sequence
            camera_dict[camera_names[i]]["Calibration_data"]["camera_matrix"], 
            camera_dict[camera_names[i]]["Calibration_data"]["distortion_coefficients"], 
            process_freq=frame_rate, 
            output_json_filename=None,
            save_to_file=False)
        camera_dict[camera_names[i]]["QR_pose_info"] = qr_pose_info
        print("----------------------------------------------")
        print(f"QR pose information for \033[92m{camera_names[i]}\033[0m obtained successfully: {qr_pose_info['frame_number']} frames.")
        for key, value in qr_pose_info["occlusion_frame_number"].items():
            print(f"QR label: {key}, occlusion frame number: \033[91m{value}\033[0m")
        print("----------------------------------------------")
    
    ##### Debug use #####
    if debug:
        # save the camera_dict here as npy file
        np.save(f"data/{test_name}/camera_dict.npy", camera_dict)
        # save qr code pose in each camera
        for camera_name in camera_names:
            for qr_label in qr_labels:
                np.save(f'data/{test_name}/{qr_label}_in_{camera_name}.npy', camera_dict[camera_name]['QR_pose_info'][qr_label])

    # camera_dict = np.load(f"data/{test_name}/camera_dict.npy", allow_pickle=True).item()

    ###### Construct camera positions with respect to the specified QR code with pose calibration video ######
    for i in range(camera_num):
        if cam_pose_calib_video_path[i] is None:
            while True:
                try:
                    path_to_video = input(f"Enter the path to the camera pose calibration video for \033[92m{camera_names[i]}\033[0m: ")
                    with open(path_to_video):
                        break
                except FileNotFoundError:
                    print(f"File path \033[92m{path_to_video}\033[0m not found. Please enter a valid path.")
        else:
            path_to_video = cam_pose_calib_video_path[i]
        
        camera_dict[camera_names[i]]["cam_pose_calib_video_path"] = path_to_video # record the path to the video
        print("----------------------------------------------")
        print(f"Calculating cam_pose calib QR code poses for \033[92m{camera_names[i]}\033[0m, video: {path_to_video}")
        print("Starting calculation...")
        # calculate the QR code poses for each frame
        cam_pose_calib_info = get_qr_poses_from_video(
            path_to_video,
            qr_sizes, # in meters
            qr_labels, # labels of QR code in sequence
            camera_dict[camera_names[i]]["Calibration_data"]["camera_matrix"], 
            camera_dict[camera_names[i]]["Calibration_data"]["distortion_coefficients"], 
            process_freq=frame_rate, 
            output_json_filename=None,
            save_to_file=False)
        camera_dict[camera_names[i]]["cam_calib_QR_pose_info"] = cam_pose_calib_info
        print("----------------------------------------------")
        print(f"QR pose information for \033[92m{camera_names[i]}\033[0m obtained successfully: {cam_pose_calib_info['frame_number']} frames.")
        for key, value in cam_pose_calib_info["occlusion_frame_number"].items():
            print(f"QR label: {key}, occlusion frame number: \033[91m{value}\033[0m")
        print("----------------------------------------------")
    for i in range(camera_num):
        if base_qr_label not in camera_dict[camera_names[i]]["cam_calib_QR_pose_info"]:
            raise ValueError(f"Wrong base_qr_label. Please provide a valid label.")
        qr_poses = camera_dict[camera_names[i]]["cam_calib_QR_pose_info"][base_qr_label] # transformation matrices for base QR code
        camera_pose = get_camera_pose_in_qr_frame(qr_poses) # camera pose with respect to base QR code
        camera_dict[camera_names[i]]["Camera_pose"] = camera_pose # record the camera pose
        print(f"Camera pose for \033[92m{camera_names[i]}\033[0m: \n{camera_pose}")

   ###### Calculate QR code poses with respect to new camera poses ######
    for i in range(camera_num):
        qr_poses = camera_dict[camera_names[i]]["QR_pose_info"]
        for label in qr_labels:
            if label != base_qr_label: # for all QR codes except the base QR code
                qr_poses_new = []
                for qr_pose in qr_poses[label]:
                    if np.array_equal(qr_pose, np.identity(4)):
                        qr_pose_new = np.array(qr_pose)
                    else:
                        qr_pose_new = np.dot(camera_dict[camera_names[i]]["Camera_pose"], qr_pose)
                    qr_poses_new.append(qr_pose_new.tolist())
                camera_dict[camera_names[i]]["QR_pose_info"][label] = qr_poses_new # update the QR pose information
    
    ###### Exclude the base QR code poses from the QR code poses ######
    for i in range(camera_num):
        camera_dict[camera_names[i]]["QR_pose_info"].pop(base_qr_label)
    qr_sizes.remove(qr_sizes[qr_labels.index(base_qr_label)])
    qr_labels.remove(base_qr_label)

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
    max_frame_best = -np.inf # the latest best starting frame among all cameras (expect basis camera)
    frame_best = {} # the best starting frame for each camera
    for i in range(camera_num):
        if camera_names[i] != basis_camera:
            frame_len = len(camera_dict[camera_names[i]]["QR_pose_info"][qr_labels[0]]) # length of QR code poses for the current camera
            qr_poses = camera_dict[camera_names[i]]["QR_pose_info"]
            qr_poses = {label: qr_poses[label] for label in qr_labels} # extract just the QR code poses for the current camera
            min_diff = float('inf')
            for starting_frame in range(0, basis_len-frame_len+1):
                basis_qr_poses_temp =\
                    {label: basis_qr_poses[label][starting_frame:starting_frame+frame_len] for label in qr_labels} # cut the basis poses
                diff = cal_poses_diff(qr_poses, basis_qr_poses_temp) # calculate the difference between the QR code poses for this part of basis poses
                # print("----------------------------------------------")
                # print(f"Starting frame: \033[92m{starting_frame}\033[0m, Difference: \033[91m{diff}\033[0m")
                if diff < min_diff:
                    min_diff = diff
                    starting_frame_best = starting_frame # record the best starting frame
            frame_best[camera_names[i]] = starting_frame_best # record the best starting frame for this camera
            print(f"Best starting frame for camera \033[92m{camera_names[i]}\033[0m: \033[92m{starting_frame_best}\033[0m")
            if starting_frame_best > max_frame_best:
                max_frame_best = starting_frame_best

    frame_num_sync = basis_len - max_frame_best - 20 # final number of frames after synchronization for basis camera
    for i in range(camera_num):
        if camera_names[i] != basis_camera:
            frame_num = camera_dict[camera_names[i]]["QR_pose_info"]["frame_number"]
            starting_frame = max_frame_best - frame_best[camera_names[i]] # starting frame for this camera
            frame_num_sync_temp = frame_num - starting_frame # final number of frames after synchronization for this camera
            if frame_num_sync_temp < frame_num_sync:
                frame_num_sync = frame_num_sync_temp
                
    print("----------------------------------------------")
    print(f"After comparison, for basis camera \033[92m{basis_camera}\033[0m on all other cameras:")
    print(f"Best starting frame: \033[92m{max_frame_best}\033[0m")
    print(f"Final number of frames: \033[92m{frame_num_sync}\033[0m")
    print("----------------------------------------------")

    # cut all cameras to frame_num_sync
    for i in range(camera_num):
        if camera_names[i] != basis_camera: # for all cameras except the basis camera
            for label in qr_labels: # for all QR codes
                starting_frame = max_frame_best - frame_best[camera_names[i]] # starting frame for this camera
                ending_frame = starting_frame + frame_num_sync # ending frame for this camera
                camera_dict[camera_names[i]]["QR_pose_info"][label] =\
                    camera_dict[camera_names[i]]["QR_pose_info"][label][starting_frame:ending_frame]
        elif camera_names[i] == basis_camera: # for basis camera
            for label in qr_labels: # for all QR codes
                starting_frame = max_frame_best
                ending_frame = starting_frame + frame_num_sync
                camera_dict[camera_names[i]]["QR_pose_info"][label] =\
                    camera_dict[camera_names[i]]["QR_pose_info"][label][starting_frame:ending_frame]

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

    ###### calculate average pose for each timestep ######
    unified_qr_poses = {label: [] for label in qr_labels} # dictionary to store the unified QR code poses
    for label in qr_labels:
        print(label)
        print(len(camera_dict[camera_names[0]]["QR_pose_info"][label]))
        print(len(camera_dict[camera_names[1]]["QR_pose_info"][label]))
        print(len(camera_dict[camera_names[2]]["QR_pose_info"][label]))
        for timestep in range(frame_num_sync):
            qr_poses = [] # list of poses for qr code with 'label' at 'timestep'
            for i in range(camera_num):
                qr_pose = camera_dict[camera_names[i]]["QR_pose_info"][label][timestep]
                if np.array_equal(qr_pose, np.identity(4)):
                    pass # not detected
                else:
                    qr_poses.append(qr_pose) # append the valid pose to the list
            if len(qr_poses) == 0: # no valid pose in all cameras
                unified_qr_poses[label].append(np.identity(4))
            if len(qr_poses) == 1: # only one valid pose
                unified_qr_poses[label].append(np.array(qr_poses[0]))
            if len(qr_poses) >= 2: # multiple valid poses
                unified_qr_poses[label].append(ave_pose(qr_poses)) # calculate the average pose
    
    ###### detect unstable QR code poses ######
    max_trans_speed = 0.5 # maximum translational speed in m/s
    max_rot_speed = 1.57 # maximum rotational speed in rad/s
    for label in qr_labels:
        for i in range(1, frame_num_sync):
            if not np.array_equal(unified_qr_poses[label][i], np.identity(4)) and\
               not np.array_equal(unified_qr_poses[label][i-1], np.identity(4)):
                if not is_similar(unified_qr_poses[label][i-1],
                                unified_qr_poses[label][i],
                                threshold_trans=max_trans_speed/frame_rate,
                                threshold_rot=max_rot_speed/frame_rate):
                    # print(f"Unstable QR code pose detected for QR code {label} at frame {i}.")
                    unified_qr_poses[label][i] = np.identity(4)
    
    ###### interpolate all identity matrices ######
    for label in qr_labels:
        identity_idx = [] # positions of identity matrices
        for i in range(frame_num_sync):
            if np.array_equal(unified_qr_poses[label][i], np.identity(4)):
                identity_idx.append(i)
        identity_idx_grouped = consecutive_idx(identity_idx)
        for idx_group in identity_idx_grouped:
            if len(idx_group) > 1: # mupltiple identity matrices
                start_idx = idx_group[0]
                end_idx = idx_group[-1]
                if start_idx == 0: # if the first frame is identity matrix
                    for idx in range(end_idx+1):
                        unified_qr_poses[label][idx] = unified_qr_poses[label][end_idx+1]
                elif end_idx == frame_num_sync-1: # if the last frame is identity matrix
                    for idx in range(start_idx, frame_num_sync):
                        unified_qr_poses[label][idx] = unified_qr_poses[label][start_idx-1]
                elif (len(idx_group) < 40 and label != '2') or label == '2':
                    interpolated_transforms = interpolate_transform(unified_qr_poses[label][start_idx-1], unified_qr_poses[label][end_idx+1], len(idx_group))
                    for idx in range(len(idx_group)):
                        unified_qr_poses[label][idx_group[idx]] = interpolated_transforms[idx]
                elif len(idx_group) >= 40 and label != '2': # if the gap is too large, do not interpolate
                    pass
            elif len(idx_group) == 1: # just one identity matrix
                if idx_group[0] == 0: # if the first frame is identity matrix
                    unified_qr_poses[label][0] = unified_qr_poses[label][1]
                elif idx_group[0] == frame_num_sync-1: # if the last frame is identity matrix
                    unified_qr_poses[label][frame_num_sync-1] = unified_qr_poses[label][frame_num_sync-2]
                else:
                    unified_qr_poses[label][idx_group[0]] = ave_pose([unified_qr_poses[label][idx_group[0]-1], unified_qr_poses[label][idx_group[0]+1]])

    if final_pose_path is None:
        final_pose_path = input("Enter the path to save the final poses dictionary: ")
        if final_pose_path[-4:] != ".npy":
            final_pose_path += ".npy"
    np.save(final_pose_path, unified_qr_poses)
    print(f"Final poses saved to {final_pose_path}")
    print("----------------------------------------------")
    # print(unified_qr_poses['2'][0:10]) # 10 transformation matrices for QR code 2

if __name__ == "__main__":
    main(calib_video_path = [None, None, None], 
         calib_data_path = ["configs/camera_calibration_iphone.json", "configs/camera_calibration_samsung.json", "configs/camera_calibration_ashwin_iphone.json"], 
         final_pose_path="data/0422_test/unified_poses.npy",
         calib_chessboard_size = (8,6), 
         calib_chessboard_square_size = 0.0245,
         cam_pose_calib_video_path = ["/home/object814/Videos/0422_exp/right_calib_pose.MOV", "/home/object814/Videos/0422_exp/left_calib_pose.mp4", "/home/object814/Videos/0422_exp/back_calib_pose.MOV"],
         operational_video_path = ["/home/object814/Videos/0422_exp/right.MOV", "/home/object814/Videos/0422_exp/left.mp4", "/home/object814/Videos/0422_exp/back.MOV"], 
         camera_num = 3, 
         camera_names = ["right", "left", "back"], 
         qr_labels = ["1", "2", "3", "4"], 
         qr_sizes = [0.163, 0.076, 0.076, 0.076], 
         base_qr_label = "1",
         test_name="0422_test",
         frame_rate = 20,
         debug=True)