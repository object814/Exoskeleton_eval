'''
Calculation of the accuracy of the qr code pose:
the Vicon markers should be placed at each edge of the qr code, to compare the difference

Parameters that need to obtain for optimization:
    the transformation matrix between qr code frames and the frames of the links of the exoskeleton
    * only needs the transformation matrix between qr codes and links of the exoskeleton that are rigidly connected

Then, the following steps can be done to obtain the joint angles of the exoskeleton:
    use the kinpy library to load the urdf file and initialize the forward kinematic model (https://pypi.org/project/kinpy/)
    use the calculated qr code frame poses to calculate the corresponding links poses
    use the forward kinematics model to do optimization on joint values based on gradient descent, to minimize the difference between the calculated link poses and the actual link poses
'''