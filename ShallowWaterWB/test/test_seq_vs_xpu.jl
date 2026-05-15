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

using Test
using Random


include("../../src/solvers.jl")
print("imported modules\n")

# gravity
const g = 1.0
## unit testing forces computation from 2d swe sequential solver with kernel fluxes  ##

@testset "cross unit testing Forces computation" begin

    # genarete Random input data
    lx = 10.0
    ly = 10.0
    nx = 100
    ny = 100
    S_r = [SVector(rand(), rand(), rand()) for _ in 1:nx, _ in 1:ny]
    Sᴸ_r = zeros(SVector{3}, nx - 1, ny)
    Sᴿ_r = zeros(SVector{3}, nx - 1, ny)
    Sᵀ_r = zeros(SVector{3}, nx, ny - 1)
    Sᴮ_r = zeros(SVector{3}, nx, ny - 1)
    F_r = zeros(SVector{3}, nx - 1, ny)
    G_r = zeros(SVector{3}, nx, ny - 1) 

    F₁ = zeros(nx - 1, ny)
    F₂ = zeros(nx - 1, ny)
    F₃ = zeros(nx - 1, ny)
    G₁ = zeros(nx, ny - 1)
    G₂ = zeros(nx, ny - 1)
    G₃ = zeros(nx, ny - 1)

    F₁_wb = zeros(nx - 1, ny)
    F₂_wb = zeros(nx - 1, ny)
    F₃_wb = zeros(nx - 1, ny)
    G₁_wb = zeros(nx, ny - 1)
    G₂_wb = zeros(nx, ny - 1)
    G₃_wb = zeros(nx, ny - 1)


    h  = zeros(nx, ny)
    hu = zeros(nx, ny)
    hv = zeros(nx, ny)

    h_wb = zeros(nx, ny)
    hu_wb = zeros(nx, ny)
    hv_wb = zeros(nx, ny)

    for i in 1:nx, j in 1:ny
        h[i, j] = S_r[i, j][1]
        hu[i, j] = S_r[i, j][2]
        hv[i, j] = S_r[i, j][3]
        h_wb[i, j] = S_r[i, j][1]
        hu_wb[i, j] = S_r[i, j][2]
        hv_wb[i, j] = S_r[i, j][3]
    end

    max_speed_x = zeros(nx-1, ny)
    max_speed_y = zeros(nx, ny-1)

    for i in 1:nx-1, j in 1:ny-1
        max_speed_x[i, j] = max(λx(S_r[i, j]), λx(S_r[i+1, j]))
        max_speed_y[i, j] = max(λy(S_r[i, j]), λy(S_r[i, j+1]))
    end

    dx = lx / (nx - 1)
    dy = ly / (ny - 1)

    z = zeros(nx, ny)
    dzdx = zeros(nx, ny)
    dzdy = zeros(nx, ny)

    vel_eps = min(dx, dy)^4

    solve_z_2d(S_r, Sᴸ_r, Sᴿ_r, Sᴮ_r, Sᵀ_r, F_r, G_r, dx, dy, dzdx, dzdy, nx, ny)
    @parallel compute_1st_2nd_and_3th_flux!(F₁, F₂, F₃, G₁, G₂, G₃, hu, hv, h, g, max_speed_x, max_speed_y)

     # test for 10 random indixes the fluxes computed by the kernel and the sequential solver are approximately equal
     for _ in 1:10
        ix = rand(1:nx-1)
        iy = rand(1:ny-1)
        @test isapprox(F₁[ix, iy], F_r[ix, iy][1]; rtol=1e-5, atol=1e-8)
        @test isapprox(F₂[ix, iy], F_r[ix, iy][2]; rtol=1e-5, atol=1e-8)
        @test isapprox(F₃[ix, iy], F_r[ix, iy][3]; rtol=1e-5, atol=1e-8)
        @test isapprox(G₁[ix, iy], G_r[ix, iy][1]; rtol=1e-5, atol=1e-8)
        @test isapprox(G₂[ix, iy], G_r[ix, iy][2]; rtol=1e-5, atol=1e-8)
        @test isapprox(G₃[ix, iy], G_r[ix, iy][3]; rtol=1e-5, atol=1e-8)
     end



end