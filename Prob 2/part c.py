import os
import numpy as np
import matplotlib.pyplot as plt

# Set directories
input_dir = "Local_density_of_states_near_band_edge"
analysis_dir = "local_density_analysis"

# Create output directory if it doesn’t exist
os.makedirs(analysis_dir, exist_ok=True)

# Function to read LDOS data
def read_ldos_file(file_path):
    return np.loadtxt(file_path, delimiter=",")  # Handle CSV format

# Function to analyze a sub-region (adjust region as needed)
def analyze_subregion(data, region=(10, 20, 10, 20)):
    """Compute the average LDOS in a subregion."""
    x_start, x_end, y_start, y_end = region
    subregion = data[y_start:y_end, x_start:x_end]
    return np.mean(subregion)

# Define the subregion bounds (adjust as needed)
subregion_bounds = (10, 20, 10, 20)

# Store subregion averages for each file
subregion_averages = []

# Process all LDOS files in the local directory
for file in sorted(os.listdir(input_dir)):  # Sorting ensures correct order
    if file.endswith(".txt"):
        file_path = os.path.join(input_dir, file)
        try:
            data = read_ldos_file(file_path)
            avg_ldos = analyze_subregion(data, subregion_bounds)
            subregion_averages.append((file, avg_ldos))
            print(f" Processed {file}, Avg LDOS: {avg_ldos:.6f}")
        except Exception as e:
            print(f" Error processing {file}: {e}")

# Sort data by filename index
subregion_averages.sort()

# Extract filenames and average values
filenames, avg_values = zip(*subregion_averages)

# Plot the subregion analysis trend
plt.figure(figsize=(7, 5))
plt.plot(range(len(avg_values)), avg_values, marker="o", linestyle="-", color="blue")
plt.xlabel("File Index")
plt.ylabel("Average LDOS in Subregion")
plt.title("Subregion LDOS Trend")
plt.grid()

# Save the trend plot
trend_plot_path = os.path.join(analysis_dir, "subregion_trend.png")
plt.savefig(trend_plot_path)
plt.show()

print(" Subregion analysis completed. Check 'local_density_analysis' folder for trend plot.")
