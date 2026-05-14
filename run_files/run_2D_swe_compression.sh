#!/bin/bash -l
#SBATCH --account=julia-gpu-course2026-ethz
#SBATCH --job-name="compressing"
#SBATCH --output=out/compression.%j.o
#SBATCH --error=out/compression.%j.e
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --uenv julia/25.5:v1 --view=juliaup julia --project src/xpu/output_compression.jl HD docs/frames/frames_topography 
