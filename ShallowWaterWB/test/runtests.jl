# NOTE: This file contains many parts that are copied from the file runtests.jl from the Package ParallelStencil.jl.
# push!(LOAD_PATH, "../src")
using ShallowWaterWB

function runtests()
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    println("Using Julia executable: $exename")
    testdir = pwd()
    println("Test directory: $testdir")

    printstyled("Testing SWE.jl\n"; bold=true, color=:white)

    run(`$exename --project -O3 --startup-file=no -t 8 $(joinpath(testdir, "test_1d_reference.jl"))`)
    run(`$exename --project -O3 --startup-file=no -t 8 $(joinpath(testdir, "test_seq_vs_xpu.jl"))`)

    return
end

runtests()