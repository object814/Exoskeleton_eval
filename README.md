# Exoskeleton_eval

A brief description of the Exoskeleton_eval project.

## Environment setup

We use conda to manage the Python(3.8) environment.

1. Install pybullet: `conda install conda-forge::pybullet==3.25` (version 3.25)
2. Install OpenCV: `pip install opencv-python==4.9.0.80` (version 4.9.0)
3. Install Open3d:
    - `conda install open3d-admin::open3d==0.15.1` (version 0.15.1)
    - `pip install scikit-learn matplotlib pandas plyfile tqdm addict`
4. Other pip dependencies:
    - pip install qrcode trimesh reportlab kinpy

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

## Environment setup for keypoint detection

We use DeepLabCut for 2D keypoint detection on both exoskeleton and *human body (TODO)

run `git submodule update --init --recursive` in the repo to clone the DeepLabCut into 3rdparty

Then run: `conda env update -f 3rdparty/DEEPLABCUT.yaml` to merge the environment into your current conda environment

For full experience of DeelLabCut, run: `pip install 'deeplabcut[gui,tf]'` 

run `conda activate DEEPLABCUT` and you are free to use the deeplabcut for keypoints tracktion

run `python -m deeplabcut` in DEEPLABCUT env to use the GUI

### Trouble shooting
In our case (Ubuntu20.04) we encpuntered:

`ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found (required by /home/YOUR_USER_NAME/anaconda3/envs/DEEPLABCUT/lib/python3.9/site-packages/zmq/backend/cython/../../../../../libzmq.so.5)`

Soulution:

`sudo rm /lib/x86_64-linux-gnu/libstdc++.so.6`
`sudo cp /home/YOUR_USER_NAME/anaconda3/envs/DEEPLABCUT/lib/libstdc++.so.6 /lib/x86_64-linux-gnu/`