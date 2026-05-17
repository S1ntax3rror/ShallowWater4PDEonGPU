#!/bin/bash

# Define the constant LOCAL domain size per GPU

LOCAL_NX=2000
LOCAL_NY=2000

# Ensure output directory exists
mkdir -p out

echo "Starting Weak Scaling Job Submissions..."
echo "========================================="

# Loop over the desired number of GPUs
for gpus in 1 2 4 8 16 32 64; do

    # Determine the 2D MPI topology (px * py = gpus)
    if [ $gpus -eq 1 ]; then px=1; py=1;
    elif [ $gpus -eq 2 ]; then px=2; py=1;
    elif [ $gpus -eq 4 ]; then px=2; py=2;
    elif [ $gpus -eq 8 ]; then px=4; py=2;
    elif [ $gpus -eq 16 ]; then px=4; py=4;
    elif [ $gpus -eq 32 ]; then px=8; py=4;
    elif [ $gpus -eq 64 ]; then px=8; py=8;
    fi

    #Calculate the required GLOBAL domain size for this topology
    GLOBAL_NX=$(( LOCAL_NX * px ))
    GLOBAL_NY=$(( LOCAL_NY * py ))

    # Calculate SLURM node and task distribution (Max 4 GPUs per node)
    if [ $gpus -lt 4 ]; then
        nodes=1
        tasks_per_node=$gpus
    else
        nodes=$(( gpus / 4 ))
        tasks_per_node=4
    fi

    job_name="weak_${gpus}gpu"
    script_name="submit_${gpus}gpu.slurm"

    echo "Preparing $job_name: $nodes Node(s), $gpus GPUs, Global Grid: ${GLOBAL_NX}x${GLOBAL_NY}"

    # Generate the SLURM batch script
    cat <<EOF > $script_name
#!/bin/bash -l
#SBATCH --account=julia-gpu-course2026-ethz
#SBATCH --job-name="$job_name"
#SBATCH --output=out/${job_name}.%j.o
#SBATCH --error=out/${job_name}.%j.e
#SBATCH --time=00:30:00
#SBATCH --nodes=$nodes
#SBATCH --ntasks=$gpus
#SBATCH --ntasks-per-node=$tasks_per_node
#SBATCH --gpus-per-task=1

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1
export JULIA_CUDA_USE_COMPAT=false

# Pass the dynamically calculated global dimensions to the Julia script
srun --uenv julia/25.5:v1 --view=juliaup julia --project src/xpu/2d_swe_multi_xpu_wb.jl --nx $GLOBAL_NX --ny $GLOBAL_NY --dt_multiplier 1
srun --uenv julia/25.5:v1 --view=juliaup julia --project src/xpu/2d_swe_multi_xpu_wb.jl --nx $GLOBAL_NX --ny $GLOBAL_NY --dt_multiplier 1
srun --uenv julia/25.5:v1 --view=juliaup julia --project src/xpu/2d_swe_multi_xpu_wb.jl --nx $GLOBAL_NX --ny $GLOBAL_NY --dt_multiplier 1
EOF

    # Submit the generated script
    sbatch $script_name
    
    rm $script_name 

done

echo "========================================="
echo "All weak scaling jobs submitted!"