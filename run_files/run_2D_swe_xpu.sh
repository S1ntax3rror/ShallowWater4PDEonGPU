#!/bin/bash -l
#SBATCH --account=julia-gpu-course2026-ethz
#SBATCH --job-name="swe_xpu"
#SBATCH --output=swe_xpu.%j.o
#SBATCH --error=swe_xpu.%j.e
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

srun --uenv julia/25.5:v1 --view=juliaup julia --project src/xpu/2d_swe_xpu.jl