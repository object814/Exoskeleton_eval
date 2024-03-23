import numpy as np

class KeypointOptimizer:
    def __init__(self, initial_positions, constraints):
        """
        Initialize the optimizer with keypoints and constraints.
        
        Parameters:
        - initial_positions: numpy array of shape (n_keypoints, 3) containing initial positions.
        - constraints: dictionary where keys are tuples of keypoints (i, j) and values are the distances.
        """
        self.initial_positions = np.array(initial_positions, dtype=np.float64)
        self.positions = np.copy(self.initial_positions)
        self.constraints = constraints
        
    def distance(self, i, j):
        """Calculate the Euclidean distance between keypoints i and j."""
        return np.linalg.norm(self.positions[i] - self.positions[j])
    
    def objective_function(self):
        """Calculate the total error based on the distance constraints."""
        error = 0
        for (i, j), distance in self.constraints.items():
            error += (self.distance(i, j) - distance)**2
        return error
    
    def gradient_descent_step(self, learning_rate=0.01):
        """
        Perform one step of gradient descent on the positions of the keypoints.
        
        Parameters:
        - learning_rate: The step size to use for the gradient descent update.
        """
        gradients = np.zeros_like(self.positions)
        
        for (i, j), distance in self.constraints.items():
            direction = self.positions[i] - self.positions[j]
            current_distance = self.distance(i, j)
            if current_distance == 0:
                continue  # Avoid division by zero
            gradient = 2 * (current_distance - distance) * (direction / current_distance)
            gradients[i] += gradient
            gradients[j] -= gradient
        
        self.positions -= learning_rate * gradients
    
    def optimize(self, iterations=1000, learning_rate=0.01):
        """Optimize the positions of the keypoints."""
        for _ in range(iterations):
            self.gradient_descent_step(learning_rate)
            
        final_error = self.objective_function()
        return self.positions, final_error
    
    def movement_metrics(self):
        """Calculate movement metrics for each keypoint."""
        position_differences = self.positions - self.initial_positions
        movement_distances = np.linalg.norm(position_differences, axis=1)
        return position_differences, movement_distances

# Example usage
# initial_positions = [
#     [0, 0, 0],   # Key point 1
#     [100, 0, 0], # Key point 2
#     [0, 100, 0], # Key point 3
#     [0, 180, 0], # Key point 4
#     [0, 300, 0], # Key point 5
# ]
initial_positions = [
    [1, 2, 0],   # Key point 1
    [100, 1, 1], # Key point 2
    [2, 100, 0], # Key point 3
    [0, 179, 1.5], # Key point 4
    [-2, 301, 0.1], # Key point 5
]

constraints = {
    (0, 1): 100, # Distance between keypoints 1 and 2
    (2, 3): 80,  # Distance between keypoints 3 and 4
    (3, 4): 120, # Distance between keypoints 4 and 5
}

optimizer = KeypointOptimizer(initial_positions, constraints)
optimized_positions, final_error = optimizer.optimize()

position_differences, movement_distances = optimizer.movement_metrics()

print("Optimized Positions:", optimized_positions)
print("Final Error:", final_error)
print("Position Differences (Vector Form):", position_differences)
print("Movement Distances (Euclidean Distance):", movement_distances)
