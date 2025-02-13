import os
import numpy as np
import matplotlib.pyplot as plt
import requests
import io

# GitHub repository containing LDOS data
github_repo_url = "https://github.com/Physics-129AL/Local_density_of_states_near_band_edges/raw/main/"

# Local directory inside your repo to save heatmaps
heatmap_dir = "local_density_heatmaps"
os.makedirs(heatmap_dir, exist_ok=True)

# List of LDOS files (You might need to get filenames manually if they are not indexed)
file_list = ["local_density_of_states_for_level_0.txt,local_density_of_states_for_level_1.txt, local_density_of_states_for_level_2.txt, local_density_of_states_for_level_3.txt, local_density_of_states_for_level_4.txt, local_density_of_states_for_level_5.txt, local_density_of_states_for_level_6.txt, local_density_of_states_for_level_7.txt, local_density_of_states_for_level_8.txt, local_density_of_states_for_level_9.txt, local_density_of_states_for_level_10.txt "]  

# Function to download and read LDOS data
def fetch_ldos_data(filename):
    file_url = github_repo_url + filename
    response = requests.get(file_url)
    if response.status_code == 200:
        data = np.loadtxt(io.StringIO(response.text))
        return data
    else:
        print(f"Failed to download {filename}")
        return None

# Function to generate a heatmap
def generate_heatmap(data, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap="inferno", origin="lower")
    plt.colorbar(label="LDOS Intensity")
    plt.title(f"Heatmap of {filename}")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Save the heatmap in your repository
    output_path = os.path.join(heatmap_dir, f"{filename}_heatmap.png")
    plt.savefig(output_path)
    plt.close()

# Process each LDOS file
for file in file_list:
    data = fetch_ldos_data(file)
    if data is not None:
        generate_heatmap(data, file)

print("Heatmaps generated and saved in your repository!")