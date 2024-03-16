import trimesh
import numpy as np

# Cube dimensions
width = 0.119  # meters
height = 0.119  # meters
depth = 0.01  # meters

# Create a box mesh
mesh = trimesh.creation.box(extents=[width, height, depth])

# Save the mesh to a file (OBJ format)
mesh.export('assets/plate_mesh.obj')
