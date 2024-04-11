import cv2
import numpy as np
import pyboof as pb

def visualize_micro_qr_codes_in_video(video_path, scale_factor=0.5):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create a Micro QR Code detector using PyBoof (BoofCV)
    detector = pb.FactoryFiducial(np.uint8).microqr()

    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert the grayscale frame to a BoofCV compatible image
        boof_image = pb.ndarray_to_boof(gray_frame)

        # Detect Micro QR codes
        detector.detect(boof_image)

        # Iterate through detected Micro QR codes
        for qr in detector.detections:
            # Convert the polygon points to a numpy array
            polygon = np.array(qr.bounds.convert_tuple(), dtype=np.int32)
            
            # Draw the bounding polygon
            cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Labels for each corner
            corner_labels = ['1: UL', '2: UR', '3: DR', '4: DL']  # Adjust based on actual order

            # Label each corner
            for i, corner in enumerate(polygon):
                print(tuple(corner))
                cv2.putText(frame, corner_labels[i], tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Put the Micro QR code's data as text near the first corner
            cv2.putText(frame, qr.message, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Resize the frame
        width = int(frame.shape[1] * scale_factor)
        height = int(frame.shape[0] * scale_factor)
        resized_frame = cv2.resize(frame, (width, height))

        # Show the resized frame
        cv2.imshow("Micro QR Code Detection", resized_frame)

        # Wait for the Enter key to be pressed to advance to the next frame
        key = cv2.waitKey(0)
        if key == 13:  # 13 is the Enter Key
            continue  # Move to the next frame
        else:
            break  # Exit loop if any other key is pressed

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = '/home/object814/Videos/iphone_test_2.MOV'
visualize_micro_qr_codes_in_video(video_path, scale_factor=0.5)  # Adjust scale_factor as needed
