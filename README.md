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
    - pip install qrcode trimesh reportlab kinpy pyboof segno

## Running the code

Please run the scripts with the terminal under the root directory (e.g., `python scripts/your_code.py`).

## Overall Pipeline

1. Customize preparation for your exoskeleton:
    - for every rigid body of the exoskeleton, at least one QR code shoud be used to represent its pose
    - generate micro QR codes with utils/micro_qrcode_generation.py, modify the parameters in the script directly according to your need
    - print out the QR codes in a size suitable for you, measure the size of the QR codes and attach to the exoskeleton

2. URDF augmentation:
    - add virtual links to the exoskeleton URDF, according to where QR codes are attached to the exoskeleton in real world

3. Before running the main calculation code, do the video recording first, including:
    - camera pose calibration video for each camera
    - operation video for each camera

4. Run main calculation code:
    python scripts/get_qr_pose.py

    change the parameters of the main function input directly in the script

5. Run pose estimation code:
    python scripts/get_exo_pose.py

    change the parameters of the main function input directly in the script

## Problem Solving

`ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found (required by /home/YOUR_USER_NAME/anaconda3/envs/DEEPLABCUT/lib/python3.9/site-packages/zmq/backend/cython/../../../../../libzmq.so.5)`

Soulution:

`sudo rm /lib/x86_64-linux-gnu/libstdc++.so.6`
`sudo cp /home/YOUR_USER_NAME/anaconda3/envs/DEEPLABCUT/lib/libstdc++.so.6 /lib/x86_64-linux-gnu/`