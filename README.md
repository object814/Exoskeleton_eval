# Exoskeleton_eval

A brief description of the Exoskeleton_eval project.

## Environment setup

We use conda to manage the Python environment.

1. Install pybullet: `conda install conda-forge::pybullet` (version 3.25)
2. Install OpenCV: `pip install opencv-python` (version 4.9.0)
3. Install Open3d:
    - `conda install open3d-admin::open3d` (version 0.15.1)
    - `pip install scikit-learn matplotlib pandas plyfile tqdm addict`

## Running the code

Please run the scripts with the terminal under the root directory (e.g., `python scripts/your_code.py`).

## Camera Calibration

Before you perform QR code tracking, you need to calibrate your device.

**If you are using a device (e.g., your smartphone) that can change intrinsic parameters (e.g., focus length, image size, etc.), please ensure that the calibration and camera configuration process are consistent.**

1. Print out a chessboard pattern for calibration.
2. Measure the size of the square and your pattern size.
3. Record a video using your device, ensuring that the chessboard is visible at all times and from various angles.
4. Run the following command in your terminal:
    ```console
    python scripts/camera_calibration.py your_video_file.mp4 9 6 25
    # chessboard with 9*6 grid size and 25 mm
    ```
