import cv2
import numpy as np
import json
import pyboof as pb

def get_qr_center_positions(frame, detector, camera_matrix, dist_coeffs):
    """
    Detects Micro QR codes in the frame and extracts their center positions.

    Args:
        frame (numpy.ndarray): The frame from the video.
        detector (pyboof): Initialized PyBoof QR detector.

    Returns:
        list: List of center positions of each detected QR code.
    """
    # Convert the frame into grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the input image to PyBoof format
    boof_image = pb.ndarray_to_boof(gray_frame)

    # Detect Micro QR codes
    detector.detect(boof_image)

    centers = []
    for qr in detector.detections:
        image_points = np.array(qr.bounds.convert_tuple(), dtype=np.float32)

        qr_size = 0.076
        object_points = np.array([
            [-qr_size/2, qr_size/2, 0],
            [qr_size/2, qr_size/2, 0],
            [qr_size/2, -qr_size/2, 0],
            [-qr_size/2, -qr_size/2, 0]
        ], dtype=np.float32)

        # SolvePnP to find the pose
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if retval:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec[:, 0]
            centers.append(T[:3, 3])
    
    return centers

def main():
    video_path = '/home/object814/Videos/precision_test/4m.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")

    detector = pb.FactoryFiducial(np.uint8).microqr()  # Initialize the PyBoof Micro QR code detector

    z_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the center positions of detected Micro QR codes
        path_to_calib_data = 'configs/camera_calibration_samsung.json'
        with open(path_to_calib_data, "r") as f:
                data = json.load(f)
                camera_matrix = np.array(data["camera_matrix"]) # record the calibration data
                dist_coeffs = np.array(data["distortion_coefficients"]) # record the calibration data
        centers = get_qr_center_positions(frame, detector, camera_matrix, dist_coeffs)
        if len(centers) == 0:
            continue
        else:
            print(centers[0])
            z_list.append(centers[0][2])
        print("Detected Micro QR code centers:", centers)

        # Display the frame
        # width = int(frame.shape[1] * 0.5)
        # height = int(frame.shape[0] * 0.5)
        # resized_frame = cv2.resize(frame, (width, height))

        # Show the resized frame
        # cv2.imshow("Micro QR Code Detection", resized_frame)
        # if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to quit the loop
        #     break

    cap.release()
    cv2.destroyAllWindows()

    # calculate the mean, std, and variance of z_list
    z_list = np.array(z_list)
    z_mean = np.mean(z_list)
    z_std = np.std(z_list)
    z_var = np.var(z_list)
    print("Mean:", z_mean)
    print("Standard Deviation:", z_std)
    print("Variance:", z_var)


if __name__ == "__main__":
    main()
