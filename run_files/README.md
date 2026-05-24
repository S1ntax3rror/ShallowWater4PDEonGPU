# RUNFILES:

Make sure to edit them first to correctly set flags if nescessary and specify the out directory (sometimes in the julia file). Also change the scale as all scripts contain scales for CPU. GPU can run at much larger scale.

Use them on the via 

```
sbatch run_files/runfile-name
```

Note that you need a running julia uenv as described here: https://pde-on-gpu.vaw.ethz.ch/software_install/

## compression
Specify the correct folder in the run_2D_swe_compression.sh script and it will compress all array output in that file to a single tar.gz with seleced resolution e.g. HD

## run_2D_swe_xpu
Runs the solver that does not conserve the steady state on xpu

## run_2D_swe_multi_xpu
Runs the solver that does not conserve the steady state on multi xpu

## run_2D_swe_xpu_wb
Runs the solver that does conserve the steady state on xpu

## run_2D_swe_multi_xpu_wb
Runs the solver that does conserve the steady state on multi xpu

## weak_scaling
Launch with 
```
bash run_files/weak_scaling.sh
```
Launches a the jobs required to to a weak_scaling analysis.
