import numpy as np
import matplotlib.pyplot as plt

# Original data
ranges = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4])
means = np.array([1.0697877768011859, 1.5828762201321556, 2.1088368232461834, 
                  2.6293780646612896, 3.1582673852200918, 3.612297739826958, 4.070840501932619])
std_devs = np.array([0.004945380914092894, 0.005853630625598277, 0.0070739205356370165, 
                     0.012562025249482018, 0.023427626810758745, 0.04747518227505281, 0.11394100892840477])

# Normalize the data
system_error = means[0] - 1  # error from the ideal 1m mean
normalized_means = means - system_error

# Analysis for range 1.5m to 4m
selected_ranges = ranges[1:]  # Exclude 1m range
selected_means = normalized_means[1:]
selected_std_devs = std_devs[1:]

# Plotting the means and standard deviations with different colors
plt.figure(figsize=(10, 5))
plt.errorbar(selected_ranges, selected_means, yerr=selected_std_devs, fmt='-o', ecolor='red', capsize=5, elinewidth=2, capthick=2, color='blue', markersize=5)
plt.title('Normalized QR Code Position Estimations (1.5m to 4m)')
plt.xlabel('Distance (m)')
plt.ylabel('Estimated Distance (m)')
plt.grid(True)

# Annotating the means and standard deviations
for i, (x, y, yerr) in enumerate(zip(selected_ranges, selected_means, selected_std_devs)):
    plt.annotate(f'{y:.3f}±{yerr:.3f}', (x, y+0.1), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

plt.show()
