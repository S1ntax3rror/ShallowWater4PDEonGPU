
using Test
using Random
using StaticArrays

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
import ParallelStencil: @reset_parallel_stencil

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=false)
    @info "threads" Threads.nthreads()
end


include("../../src/solvers.jl")
print("imported modules\n")

# gravity
const g = 1.0
## unit testing from reference to 1d swe with topography ##

@testset "verifying solving one time step" begin

    # genarete Random input data
    lx = 10.0
    nx = 100
    S_r = [SVector(rand(), rand()) for _ in 1:nx]
    Sᴸ_r = zeros(SVector{2}, nx - 1)
    Sᴿ_r = zeros(SVector{2}, nx - 1)
    F_r = zeros(SVector{2}, nx - 1)

    S = copy(S_r)
    Sᴸ = copy(Sᴸ_r)
    Sᴿ = copy(Sᴿ_r)
    F = copy(F_r)

    dx = lx / (nx - 1)

    # topography flat as we only want to test the solver to refence without topography
    z = zeros(nx)
    dzdx = zeros(nx)

    # update state 

    solve_z(S, Sᴸ, Sᴿ, F, dx, dzdx, nx)
    solve(S_r, Sᴸ_r, Sᴿ_r, F_r, dx)



    # # test for 10 random indixes
     for _ in 1:10
         ix = rand(1:nx)
         @test isapprox(S[ix][1], S_r[ix][1]; rtol=1e-5, atol=1e-8)
         @test isapprox(S[ix][2], S_r[ix][2]; rtol=1e-5, atol=1e-8)
     end


end
