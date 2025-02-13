import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set directories
input_dir = "Local_density_of_states_near_band_edge"
surface_plot_dir = "local_density_surfaces"

# Create output directory if it doesn’t exist
os.makedirs(surface_plot_dir, exist_ok=True)

# Function to read LDOS data
def read_ldos_file(file_path):
    return np.loadtxt(file_path, delimiter=",")  # Handle CSV format

# Function to generate a 3D surface plot
def generate_surface_plot(data, filename):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    # Create meshgrid for x and y axes
    x = np.arange(data.shape[1])  # Columns
    y = np.arange(data.shape[0])  # Rows
    X, Y = np.meshgrid(x, y)

    # Plot surface
    ax.plot_surface(X, Y, data, cmap="viridis", edgecolor="none")
    ax.set_title(f"3D Surface Plot of {filename}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("LDOS Intensity")

    # Save the 3D plot
    output_path = os.path.join(surface_plot_dir, f"{filename}_surface.png")
    plt.savefig(output_path)
    plt.close()

# Process all LDOS files in the local directory
for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(input_dir, file)
        try:
            data = read_ldos_file(file_path)
            generate_surface_plot(data, file)
            print(f" 3D Surface plot generated for {file}")
        except Exception as e:
            print(f" Error processing {file}: {e}")

print(" All 3D surface plots saved successfully! Check  'local_density_surfaces' folder.")
