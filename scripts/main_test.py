import numpy as np
import json
import pprint
import os
import sys
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.camera_calibration import calibrate_camera_from_video
from utils.cal_poses_diff import cal_poses_diff
from utils.get_qr_pose import get_qr_poses_from_video
from utils.get_camera_pose import get_camera_pose_in_qr_frame

def is_similar(pose1, pose2, threshold_trans=0.05, threshold_rot=0.15):
    """
    Determine if two poses are similar based on translational and rotational thresholds.

    Args:
        pose1 (np.array): The first pose as a 4x4 transformation matrix.
        pose2 (np.array): The second pose as a 4x4 transformation matrix.
        threshold_trans (float): The translational threshold in meters.
        threshold_rot (float): The rotational threshold in radians.

    Returns:
        flag: True if the poses are similar, False otherwise.
    """

    pose1 = np.array(pose1)
    pose2 = np.array(pose2)

    # Extract translational components and calculate difference
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]
    translational_diff = np.linalg.norm(t1 - t2)

    # Extract rotational components and calculate geodesic distance
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    trace = np.trace(np.dot(np.transpose(R1), R2))
    theta = np.arccos(max(min((trace - 1) / 2, 1), -1))

    # Determine if the poses are similar
    if translational_diff <= threshold_trans and theta <= threshold_rot:
        return True
    else:
        return False

def interpolate_transform(start_T, end_T, steps):
    """
    Performs linear interpolation on transformation matrices.

    Args:
        start_T (np.array): The starting 4x4 transformation matrix.
        end_T (np.array): The ending 4x4 transformation matrix.
        steps (int): The number of intermediate matrices to be outputted.

    Returns:
        list: A list of interpolated 4x4 transformation matrices, including the start and end matrices.
    """
    start_T = np.array(start_T)
    end_T = np.array(end_T)

    # Initialize the list of interpolated matrices, starting with the start matrix
    interpolated_matrices = [start_T]
    
    # Calculate the step increment for translation and rotation (quaternion)
    start_translation, end_translation = start_T[:3, 3], end_T[:3, 3]
    translation_step = (end_translation - start_translation) / (steps + 1)
    
    start_rotation = R.from_matrix(start_T[:3, :3])
    end_rotation = R.from_matrix(end_T[:3, :3])

    # Create a spherical linear interpolation object
    key_rots = R.from_quat(np.vstack([start_rotation.as_quat(), end_rotation.as_quat()]))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)

    times = np.linspace(0, 1, steps + 2)
    interp_rots = slerp(times)
    
    for step in range(1, steps + 1):
        # Interpolate translation
        interpolated_translation = start_translation + translation_step * step
        
        # Construct the interpolated transformation matrix
        interpolated_T = np.eye(4)
        interpolated_T[:3, :3] = interp_rots[step].as_matrix()
        interpolated_T[:3, 3] = interpolated_translation
        
        # Append the interpolated transformation matrix to the list
        interpolated_matrices.append(interpolated_T)
    
    # Append the end matrix to the list of interpolated matrices
    interpolated_matrices.append(end_T)

    interpolated_matrices.pop(0) # remove the first matrix
    interpolated_matrices.pop(-1) # remove the last matrix
     
    return interpolated_matrices
    
def ave_rotation(R_list):
    """
    Computes the average rotation matrix from a list of rotation matrices.
    
    Args:
        rotations (list of np.array): A list of 3x3 rotation matrices.
    
    Returns:
        np.array: The average rotation matrix.
    """
    # Step a: Compute the 'mean matrix' M of the rotations
    M = sum(R_list) / len(R_list)
    
    # Step b: Take the SVD of that mean matrix M
    U, _, Vt = np.linalg.svd(M)
    
    # Step c: Compute the rotation closest to M
    Q = np.dot(U, Vt)
    
    # Ensure Q is a proper rotation matrix by checking its determinant
    # and correcting if necessary
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1  # Flip last column of U if determinant of Q is negative
        Q = np.dot(U, Vt)
    
    return Q

def ave_pose(T_list):
    """
    Calculate the average pose of a list of poses.

    Args:
        T_list (list of np.array): A list of 4x4 transformation matrices.

    Returns:
        np.array: The average pose as a 4x4 transformation matrix.
    """

    # Extract translational components and calculate average
    ave_t = np.mean([np.array(T)[:3, 3] for T in T_list], axis=0)

    # Extract rotational components and calculate average
    ave_R = ave_rotation([np.array(T)[:3, :3] for T in T_list])

    ave_T = np.eye(4)
    ave_T[:3, :3] = ave_R
    ave_T[:3, 3] = ave_t

    return ave_T

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

def main(calibration_video_path = None,
         calibration_data_path = None,
         calib_chessboard_size = None,
         calib_chessboard_square_size = None,
         calculation_video_path = None,
         camera_num = 2, 
         camera_names = ["camera1", "camera2"],
         frame_rate = 10,
         qr_labels = ["1", "2"], 
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
    '''
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
            if calibration_video_path[i] is None:
                while True:
                    try:
                        path_to_video = input(f"Enter the path to the calibration video for \033[92m{camera_names[i]}\033[0m: ")
                        with open(path_to_video):
                            break
                    except FileNotFoundError:
                        print(f"File path \033[92m{path_to_video}\033[0m not found. Please enter a valid path.")
            else:
                path_to_video = calibration_video_path[i]
            
            print(path_to_video)
            camera_dict[camera_names[i]]["Calibration_video_path"] = path_to_video # record the path to the calibration video
            calibration_data = calibrate_camera_from_video(path_to_video, calib_chessboard_size, calib_chessboard_square_size, 30, False)
            camera_dict[camera_names[i]]["Calibration_data"] = calibration_data # record the calibration data
        else:
            if calibration_data_path[i] is None:
                while True:
                    try:
                        path_to_calib_data = input(f"Enter the path to the calibration JSON data for \033[92m{camera_names[i]}\033[0m: ")
                        with open(path_to_calib_data):
                            break
                    except FileNotFoundError:
                        print(f"File path \033[92m{path_to_calib_data}\033[0m not found. Please enter a valid path.")
            else:
                path_to_calib_data = calibration_data_path[i]

            camera_dict[camera_names[i]]["Calibration_data_path"] = path_to_calib_data # record the path to the calibration data
            with open(path_to_calib_data, "r") as f:
                data = json.load(f)
                camera_dict[camera_names[i]]["Calibration_data"]["camera_matrix"] = np.array(data["camera_matrix"]) # record the calibration data
                camera_dict[camera_names[i]]["Calibration_data"]["distortion_coefficients"] = np.array(data["distortion_coefficients"]) # record the calibration data

    ###### Calculate QR code pose for each camera ######
    for i in range(camera_num):
        if calculation_video_path[i] is None:
            while True:
                try:
                    path_to_video = input(f"Enter the path to the calculation video for \033[92m{camera_names[i]}\033[0m: ")
                    with open(path_to_video):
                        break
                except FileNotFoundError:
                    print(f"File path \033[92m{path_to_video}\033[0m not found. Please enter a valid path.")
        else:
            path_to_video = calculation_video_path[i]
        
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
        print("##############################################")
        print(f"QR pose information for \033[92m{camera_names[i]}\033[0m obtained successfully: {qr_pose_info['frame_number']} frames.")
        for key, value in qr_pose_info["occlusion_frame_number"].items():
            print(f"QR label: {key}, occlusion frame number: \033[91m{value}\033[0m")
        print("##############################################")
    
    ##### Debug use #####
    # save the camera_dict here as npy file
    # np.save("/home/object814/Workspace/Exoskeleton_eval/data/camera_dict_temp.npy", camera_dict)
    # return
    '''
    camera_dict = np.load('data/0409_test/camera_dict_0409.npy', allow_pickle=True).item()

    ###### Construct camera positions with respect to the specified QR code ######
    for i in range(camera_num):
        if base_qr_label not in camera_dict[camera_names[i]]["QR_pose_info"]:
            raise ValueError(f"Wrong base_qr_label. Please provide a valid label.")
        qr_poses = camera_dict[camera_names[i]]["QR_pose_info"][base_qr_label] # transformation matrices for base QR code
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

    np.save('/home/object814/Workspace/Exoskeleton_eval/data/0409_test/iphone_camera_in_qr1.npy', camera_dict['iphone']['Camera_pose'])
    np.save('/home/object814/Workspace/Exoskeleton_eval/data/0409_test/samsung_camera_in_qr1.npy', camera_dict['samsung']['Camera_pose'])

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
                print("----------------------------------------------")
                print(f"Starting frame: \033[92m{starting_frame}\033[0m, Difference: \033[91m{diff}\033[0m")
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

    np.save('/home/object814/Workspace/Exoskeleton_eval/data/0409_test/iphone_qr2_in_qr1.npy', camera_dict['iphone']['QR_pose_info']['2'])
    np.save('/home/object814/Workspace/Exoskeleton_eval/data/0409_test/samsung_qr2_in_qr1.npy', camera_dict['samsung']['QR_pose_info']['2'])
    
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
                    print(f"Unstable QR code pose detected for QR code {label} at frame {i}.")
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
                else:
                    interpolated_transforms = interpolate_transform(unified_qr_poses[label][start_idx-1], unified_qr_poses[label][end_idx+1], len(idx_group))
                    for idx in range(len(idx_group)):
                        unified_qr_poses[label][idx_group[idx]] = interpolated_transforms[idx]
            elif len(idx_group) == 1: # just one identity matrix
                if idx_group[0] == 0: # if the first frame is identity matrix
                    unified_qr_poses[label][0] = unified_qr_poses[label][1]
                elif idx_group[0] == frame_num_sync-1: # if the last frame is identity matrix
                    unified_qr_poses[label][frame_num_sync-1] = unified_qr_poses[label][frame_num_sync-2]
                else:
                    unified_qr_poses[label][idx_group[0]] = ave_pose([unified_qr_poses[label][idx_group[0]-1], unified_qr_poses[label][idx_group[0]+1]])

    np.save('/home/object814/Workspace/Exoskeleton_eval/data/0409_test/qr2_final_poses.npy', unified_qr_poses)   
    print(unified_qr_poses['2'][0:10]) # 10 transformation matrices for QR code 2

if __name__ == "__main__":
    main(calibration_video_path = [None, None], 
         calibration_data_path = ["configs/camera_calibration_iphone.json", "configs/camera_calibration.json"], 
         calib_chessboard_size = (8,6), 
         calib_chessboard_square_size = 0.0382, 
         calculation_video_path = ["/home/object814/Videos/iphone_1.mp4", "/home/object814/Videos/samsung_1.mp4"], 
         camera_num = 2, 
         camera_names = ["iphone", "samsung"], 
         qr_labels = ["1", "2"], 
         qr_sizes = [0.1424, 0.0755], 
         base_qr_label = "1")