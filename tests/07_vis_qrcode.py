import cv2

def visualize_qr_codes_in_video(video_path, scale_factor=0.5):
    # Create a QR Code Detector
    detector = cv2.QRCodeDetector()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Detect and decode QR codes in the frame
        retval, data, points, straight_qrcode = detector.detectAndDecodeMulti(frame)
        
        if points is not None:
            # Iterate through detected QR codes
            for i in range(len(data)):
                qr_points = points[i].astype(int)  # QR code corner points
                for j in range(4):
                    # Draw lines between the corner points
                    pt1 = tuple(qr_points[j])
                    pt2 = tuple(qr_points[(j+1) % 4])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), thickness=2)

                # Put the QR code's data as text near the first corner
                cv2.putText(frame, data[i], tuple(qr_points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Resize the frame
        width = int(frame.shape[1] * scale_factor)
        height = int(frame.shape[0] * scale_factor)
        resized_frame = cv2.resize(frame, (width, height))

        # Show the resized frame
        cv2.imshow("QR Code Detection", resized_frame)

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
video_path = 'data/execute_samsung.mp4'
visualize_qr_codes_in_video(video_path, scale_factor=0.6)  # Adjust scale_factor as needed
