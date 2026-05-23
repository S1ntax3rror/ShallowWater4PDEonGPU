import os
import matplotlib.pyplot as plt

files = os.listdir("outFiles/weak2")

timingDict = {}

for f in files:

    if "swe_xpu" in f:
        continue

    lines = open(os.path.join("outFiles/weak2", f)).readlines()

    rep = 0
    sum_timings = 0.0
    num_timings = 0
    name = f.split(".")[0].lstrip("weak_")
    timingDict[name] = {"measurements" : [], "compilePercentage" : [], "actualTime" : [], "compileTime": [],
                        "actualTimeMin": 0.0, "compileTimeAverage": 0.0, "average" : 0.0}

    for line in lines:
        if "Global domain size (including halos):" in line:
            if num_timings > 0:
                timingDict[name]["measurements"].append(sum_timings / num_timings)

            resolution = line.lstrip("Global domain size (including halos):")
            # print(resolution)

            rep += 1
            num_timings = 0
            sum_timings = 0.0

        if "seconds (" in line:
            backupline = line
            s_line = line
            timing = float(s_line.split("seconds")[0])

            if "compilation time" in line:
                if "lock conflict, " in line:
                    percentage = float(line.split("lock conflict, ")[1].split("% compilation time")[0].strip()) / 100
                else:
                    percentage = float(line.split("gc time, ")[1].split("% compilation time")[0].strip()) / 100

                timingDict[name]["compilePercentage"].append(percentage)
                timingDict[name]["compileTime"].append(percentage*timing)
                timingDict[name]["actualTime"].append(timing - percentage*timing)
            else:
                timingDict[name]["compilePercentage"].append(0)
                timingDict[name]["compileTime"].append(0)
                timingDict[name]["actualTime"].append(timing)

            sum_timings += timing
            num_timings += 1

    if num_timings > 0:
        timingDict[name]["measurements"].append(sum_timings / num_timings)

    timingDict[name]["average"] = sum(timingDict[name]["measurements"]) / len(timingDict[name]["measurements"])
    timingDict[name]["actualTimeMin"] = min(timingDict[name]["actualTime"])
    timingDict[name]["compileTimeAverage"] = sum(timingDict[name]["compileTime"]) / len(timingDict[name]["compileTime"])

# --- Plotting Section ---

if not timingDict:
    print("No timing data found. Check your file contents and paths.")
else:
    sorted_gpus = sorted(timingDict.keys(), key=lambda x: int(x.split("gpu")[0]))
    sorted_times = [timingDict[gpu]["average"] for gpu in sorted_gpus]
    sorted_actualtimes = [timingDict[gpu]["actualTimeMin"] for gpu in sorted_gpus]
    sorted_compiletimes = [timingDict[gpu]["compileTimeAverage"] for gpu in sorted_gpus]

    plt.figure(figsize=(8, 6))

    # plt.plot(sorted_gpus, sorted_times, marker='o', linestyle='-', color='b', linewidth=2, markersize=8,
    #          label='Total Time')

    first = sorted_actualtimes[0]
    normed_actualtimes = []
    for time in sorted_actualtimes:
        normed_actualtimes.append(time/first)

    plt.plot(sorted_gpus, normed_actualtimes, marker='o', linestyle='-', color='g', linewidth=2, markersize=8,
             label='Actual Time')

    # plt.plot(sorted_gpus, sorted_compiletimes, marker='o', linestyle='-', color='r', linewidth=2, markersize=8,
    #          label='Compile Time')

    # for gpu in sorted_gpus:
    #     plt.scatter(
    #         [gpu] * len(timingDict[gpu]["measurements"]),
    #         timingDict[gpu]["measurements"],
    #         alpha=0.7
    #     )

    # Plot ideal weak scaling (horizontal line based on the 1-GPU/smallest-GPU run)

    plt.xlabel('Number of GPUs', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Weak Scaling Performance', fontsize=14, fontweight='bold')

    plt.xticks(sorted_gpus)

    plt.ylim(bottom=0, top=max(normed_actualtimes) * 1.2)

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    ideal_time = 1
    plt.axhline(y=ideal_time, color='r', linestyle='--', linewidth=2, label='Ideal Weak Scaling')

    plt.savefig('weak_scaling_plot_dt.png', dpi=300)
    print("Plot saved as 'weak_scaling_dtupdated_plot_base-resolution_nx-ny2000_nt2000.png'")
    plt.show()