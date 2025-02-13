import os
import numpy as np
import matplotlib.pyplot as plt

# Set local directory where LDOS files are stored
input_dir = "Local_density_of_states_near_band_edge"
heatmap_dir = "local_density_heatmaps"

# Create output directory if it doesn’t exist
os.makedirs(heatmap_dir, exist_ok=True)

# Function to read LDOS data (Fixed for CSV format)
def read_ldos_file(file_path):
    return np.loadtxt(file_path, delimiter=",")  # Added delimiter to handle commas

# Function to generate heatmap
def generate_heatmap(data, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap="inferno", origin="lower")
    plt.colorbar(label="LDOS Intensity")
    plt.title(f"Heatmap of {filename}")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Save the heatmap in the output directory
    output_path = os.path.join(heatmap_dir, f"{filename}_heatmap.png")
    plt.savefig(output_path)
    plt.close()

# Process all LDOS files in the local directory
for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(input_dir, file)
        try:
            data = read_ldos_file(file_path)  # Use the fixed function
            generate_heatmap(data, file)
            print(f" Heatmap generated for {file}")
        except Exception as e:
            print(f" Error processing {file}: {e}")

print(" Heatmaps saved successfully! Check 'local_density_heatmaps' folder.")
