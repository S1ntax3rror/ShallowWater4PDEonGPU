import os
import matplotlib.pyplot as plt

files = os.listdir("outFiles")

timingDict = {}

for f in files:
    name = f.split("gpu")[0].lstrip("weak_")
    print(name)

    lines = open(os.path.join("outFiles", f)).readlines()

    for line in lines:
        if "Total simulation time:" in line:
            time = float(line.split("Total simulation time:")[1].rstrip(" seconds\n"))
            timingDict[int(name)] = time

# --- Plotting Section ---

if not timingDict:
    print("No timing data found. Check your file contents and paths.")
else:
    # Sort the dictionary by the number of GPUs (keys)
    sorted_gpus = sorted(timingDict.keys())
    sorted_times = [timingDict[gpu] for gpu in sorted_gpus]

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot actual execution times
    plt.plot(sorted_gpus, sorted_times, marker='o', linestyle='-', color='b', linewidth=2, markersize=8,
             label='Actual Time')

    # Plot ideal weak scaling (horizontal line based on the 1-GPU/smallest-GPU run)
    ideal_time = sorted_times[0]
    plt.axhline(y=ideal_time, color='r', linestyle='--', linewidth=2, label='Ideal Weak Scaling')

    # Formatting the plot nicely
    plt.xlabel('Number of GPUs', fontsize=12)
    plt.ylabel('Total Simulation Time (seconds)', fontsize=12)
    plt.title('Weak Scaling Performance', fontsize=14, fontweight='bold')

    # Ensure x-axis ticks match your GPU counts exactly (e.g., 1, 2, 4, 8)
    plt.xticks(sorted_gpus)

    # Set y-axis limits to start a bit below the lowest time to show the floor clearly
    plt.ylim(bottom=min(sorted_times) * 0.8, top=max(sorted_times) * 1.2)

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # Save the plot to a file and display it
    plt.savefig('weak_scaling_plot.png', dpi=300)
    print("Plot saved as 'weak_scaling_plot.png'")
    plt.show()