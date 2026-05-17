import os
import matplotlib.pyplot as plt

# Point this to the file containing your raw stdout data
filename = "outFiles/swe_xpu.3463869.o"

timingDict = {}
current_res = None

# Read the file
if os.path.exists(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
else:
    print(f"Error: Could not find {filename}")
    lines = []

# Parse the data
for line in lines:
    if "Running performance test at resolution:" in line:
        res_str = line.split("resolution:")[1].split("x")[0].strip()
        current_res = int(res_str)

        if current_res not in timingDict:
            timingDict[current_res] = {"measurements": [], "average": 0.0}

    elif "seconds (" in line and current_res is not None:
        timing_str = line.split("seconds")[0].strip()
        timing = float(timing_str)

        timingDict[current_res]["measurements"].append(timing)
        current_res = None

    # --- Plotting Section ---
if not timingDict:
    print("No timing data found.")
else:
    # Filter out any resolutions that didn't finish
    valid_resolutions = [res for res in timingDict.keys() if len(timingDict[res]["measurements"]) > 0]

    for res in valid_resolutions:
        timingDict[res]["average"] = sum(timingDict[res]["measurements"]) / len(timingDict[res]["measurements"])

    sorted_res = sorted(valid_resolutions)
    sorted_times = [timingDict[res]["average"] for res in sorted_res]

    plt.figure(figsize=(9, 6))
    plt.yscale('log')

    # Plot avg
    plt.plot(sorted_res, sorted_times, marker='o', linestyle='-', color='b', linewidth=2, markersize=8,
             label='Average Time')

    # Plot measurements
    for res in sorted_res:
        plt.scatter(
            [res] * len(timingDict[res]["measurements"]),
            timingDict[res]["measurements"],
            alpha=0.7,
            color='red',
            zorder=3,
            label='Individual Trials' if res == sorted_res[0] else ""
        )

    plt.xlabel('Grid Resolution (N for N x N)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Strong Scaling / Performance Scaling', fontsize=14, fontweight='bold')

    plt.xticks(sorted_res, rotation=45)

    plt.ylim(bottom=0, top=max(sorted_times) * 1.1)

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig('performance_scaling_plot.png', dpi=300)
    print("Plot saved as 'performance_scaling_plot.png'")
    plt.show()