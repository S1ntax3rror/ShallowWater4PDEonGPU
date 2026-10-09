#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
task_python="${SWE_PYTHON:-cache/swiss_dam_venv/bin/python}"
task_threads="${SWE_THREADS:-4}"

"$task_python" examples/cleuson/prepare_terrain.py \
    --dx 5 --end-time 200 --snapshot-interval 0.5 --output cache/swiss_dam/cleuson_5m
julia --project --startup-file=no -t "$task_threads" examples/cleuson/run.jl \
    cache/swiss_dam/cleuson_5m cache/swiss_dam/frames_5m
cp cache/swiss_dam/cleuson_5m/case.toml data/cleuson/case_5m.toml
cp cache/swiss_dam/frames_5m/verification.toml data/cleuson/verification_5m.toml
cp cache/swiss_dam/frames_5m/diagnostics.csv data/cleuson/diagnostics_5m.csv
"$task_python" plotting/plot_depth_3d.py \
    --case cache/swiss_dam/cleuson_5m --frames cache/swiss_dam/frames_5m \
    --output docs/animations/cleuson_dam_break_5m_3d.mp4
"$task_python" plotting/plot_depth.py \
    --case cache/swiss_dam/cleuson_5m --frames cache/swiss_dam/frames_5m \
    --output docs/animations/cleuson_dam_break_5m_2d.mp4
