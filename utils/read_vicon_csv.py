import numpy as np
from sklearn.decomposition import PCA
import csv

def extract_vicon_data(csv_file_path):
    marker_positions = {}
    
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        
        # Skip the initial descriptive lines
        next(reader)
        next(reader)
        
        # Extracting marker names
        markers_line = next(reader)
        marker_names = [marker.strip() for marker in markers_line if marker.strip()]
        marker_names = [name.split(':')[-1] for name in marker_names if ':' in name]
        
        for name in marker_names:
            marker_positions[name] = []
        
        # Skip lines for column headers and units
        next(reader)
        next(reader)
        
        # Reading each row for data
        for row in reader:
            coordinates = row[2:]  # Skipping 'Frame' and 'Sub Frame' columns
            
            for i, name in enumerate(marker_names):
                index = i * 3
                # Ensure the index does not exceed the list bounds
                if index + 2 < len(coordinates):
                    try:
                        x, y, z = float(coordinates[index]), float(coordinates[index + 1]), float(coordinates[index + 2])
                        marker_positions[name].append([x, y, z])
                    except ValueError:
                        # Handles the case for missing or invalid data points
                        continue
                else:
                    # Handle cases where there are not enough coordinates for this marker
                    continue
    
    return marker_positions

def calculate_plane_pose_from_three_points(points):
    points_array = np.array(points)
    centroid = np.mean(points_array, axis=0)
    centroid = centroid / 1000 # Convert to meters
    v1 = points_array[1] - points_array[0]
    v2 = points_array[2] - points_array[0]
    normal_vector = np.cross(v1, v2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector, centroid

def calculate_plane_pose_from_four_points(points):
    points_array = np.array(points)
    centroid = np.mean(points_array, axis=0)
    centroid = centroid / 1000 # Convert to meters
    pca = PCA(n_components=3)
    pca.fit(points_array)
    normal_vector = pca.components_[-1]
    return normal_vector, centroid

# Assume extract_vicon_data function is already defined

def calculate_pose_at_each_timestep(marker_data):
    # Prepare data structures for storing the calculated poses
    poses = {
        "trunk": [],
        "leftThigh": [],
        "rightThigh": []
    }
    
    # Calculate the number of timesteps
    timesteps = len(marker_data["trunk1"])
    
    for timestep in range(timesteps):
        # Trunk plane
        trunk_points = [marker_data["trunk" + str(i)][timestep] for i in range(1, 5)]
        normal_vector, centroid = calculate_plane_pose_from_four_points(trunk_points)
        poses["trunk"].append((normal_vector, centroid))
        
        # Left thigh plane
        leftThigh_points = [marker_data["LeftThigh" + str(i)][timestep] for i in range(1, 4)]
        normal_vector, centroid = calculate_plane_pose_from_three_points(leftThigh_points)
        poses["leftThigh"].append((normal_vector, centroid))
        
        # Right thigh plane
        rightThigh_points = [marker_data["RightThigh" + str(i)][timestep] for i in range(1, 4)]
        normal_vector, centroid = calculate_plane_pose_from_three_points(rightThigh_points)
        poses["rightThigh"].append((normal_vector, centroid))
    
    return poses

# Example usage
csv_file_path = 'data/vicon_data_0404.csv'
marker_data = extract_vicon_data(csv_file_path)
poses = calculate_pose_at_each_timestep(marker_data)

# Example of accessing the calculated poses
print("Trunk plane pose at the first timestep:", poses["trunk"][0])
print("Left thigh plane pose at the first timestep:", poses["leftThigh"][0])
print("Right thigh plane pose at the first timestep:", poses["rightThigh"][0])

# Save data
np.save('data/vicon_test/poses.npy', poses)