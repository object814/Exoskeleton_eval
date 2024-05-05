import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

# Load the data
joint_angles_data = np.load("data/exo_pose.npy")
trunk_inclination_data = np.load("data/trunk_inclination.npy")

# Setup the first plot (joint angles)
fig1, axs = plt.subplots(4, 1, figsize=(10, 8))
titles = [
    "Left Thigh Joint 1 Angle",
    "Left Thigh Joint 2 Angle",
    "Right Thigh Joint 1 Angle",
    "Right Thigh Joint 2 Angle"
]
columns_to_plot = [0, 1, 3, 4]
time_seconds = joint_angles_data.shape[0] / 20
y_limits = [(joint_angles_data[:, col].min(), joint_angles_data[:, col].max()) for col in columns_to_plot]

def animate_joint_angles(i):
    x_values = np.linspace(0, i / 20, i)  # Convert frame index to seconds
    for idx, ax in enumerate(axs):
        ax.clear()
        ax.plot(x_values, joint_angles_data[:i, columns_to_plot[idx]])
        ax.set_title(titles[idx])
        ax.set_xlim(0, time_seconds)
        ax.set_ylim(y_limits[idx])
        ax.set_ylabel('Angle (degrees)')
        ax.set_xlabel('Time (seconds)')
    plt.tight_layout()

ani_joint_angles = FuncAnimation(fig1, animate_joint_angles, frames=len(joint_angles_data), interval=50)

# Save the joint angles animation
ani_joint_angles.save('/home/object814/Videos/joint_angles_animation.mp4', writer='ffmpeg', fps=20, extra_args=['-vcodec', 'mpeg4'])

# Setup the second plot (trunk inclination)
# fig2, ax2 = plt.subplots(figsize=(10, 4))
# time_seconds_trunk = trunk_inclination_data.shape[0] / 20
# y_limits_trunk = (trunk_inclination_data.min(), trunk_inclination_data.max())

# def animate_trunk_inclination(i):
#     x_values = np.linspace(0, i / 20, i)  # Convert frame index to seconds
#     ax2.clear()
#     ax2.plot(x_values, trunk_inclination_data[:i])
#     ax2.set_title('Trunk Inclination')
#     ax2.set_xlim(0, time_seconds_trunk)
#     ax2.set_ylim(y_limits_trunk)
#     ax2.set_ylabel('Inclination (degrees)')
#     ax2.set_xlabel('Time (seconds)')
#     plt.tight_layout()

# ani_trunk_inclination = FuncAnimation(fig2, animate_trunk_inclination, frames=len(trunk_inclination_data), interval=50)

# # Save the trunk inclination animation
# ani_trunk_inclination.save('/home/object814/Videos/trunk_inclination_animation.mp4', writer='ffmpeg', fps=20, extra_args=['-vcodec', 'mpeg4'])

plt.show()
