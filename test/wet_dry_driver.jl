using Test
using Serialization

ENV["SWE_HEADLESS"] = "true"
const multi = ARGS[1] == "multi"
const solver = multi ? "2d_swe_multi_xpu_wb.jl" : "2d_swe_xpu_wb.jl"
include(joinpath(@__DIR__, "..", "src", "xpu", solver))
include("wet_dry_cases.jl")

function test_draining()
    nx, ny = 8, 7
    h = [0.01 + 0.002*mod(i+2*j, 3) for i in 1:nx, j in 1:ny]
    h[4:5, 3:4] .= 0.0
    hu = 3.0 .* h
    hv = -2.0 .* h
    z = zeros(nx, ny)
    F₁, F₂, F₃, Pₓ, sx, tx = [zeros(nx-1, ny) for _ in 1:6]
    G₁, G₂, G₃, Pᵧ, sy, ty = [zeros(nx, ny-1) for _ in 1:6]
    drain = zeros(nx, ny)
    dt = 1.0
    @parallel compute_maxspeed!(sx, sy, h, hu, hv, z, g, 1e-16)
    @parallel compute_1st_2nd_and_3th_flux!(F₁, F₂, F₃, G₁, G₂, G₃, Pₓ, Pᵧ,
                                           hu, hv, h, z, g, sx, sy, 1e-16)
    F₁[1, :] .= F₁[end, :] .= 0.0
    G₁[:, 1] .= G₁[:, end] .= 0.0
    mass = sum(h[2:end-1, 2:end-1])
    @parallel compute_draining_timestep!(drain, F₁, G₁, h, dt, 1.0, 1.0)
    @parallel compute_effective_flux_timesteps!(tx, ty, drain, F₁, G₁, dt)
    @test minimum(tx) < dt
    @test minimum(ty) < dt
    @test all(drain[1, :] .== dt)
    @parallel update_height_momentum!(h, hu, hv, F₁, G₁, F₂, F₃, G₂, G₃,
                                      tx, ty, Pₓ, Pᵧ, z, g, dt, 1.0, 1.0)
    # Check positivity and conservation before a cleanup can conceal a failure.
    @test minimum(h) >= -1e-14
    @test abs(sum(h[2:end-1, 2:end-1]) - mass) < 1e-14
    @test all(isfinite, hu) && all(isfinite, hv)

    h .= 5e-12
    @parallel dry_cell_fix!(h, hu, hv, h_eps)
    @test all(h .== 5e-12)
    @test all(iszero, hu) && all(iszero, hv)
end

function verify()
    if multi
        MPI.Init()
    end
    dims = multi ? (parse(Int, ARGS[3]), parse(Int, ARGS[4])) : (1, 1)
    results = Dict{String, Any}()
    @testset "$(multi ? "multi-XPU" : "XPU") wet/dry verification" begin
        test_draining()
        for case in wet_dry_cases()
            common = (nt=verification_steps, domain_expansion_factor=1, do_viz=false,
                      bathymetry=case.bed, initial_surface=case.surface,
                      domain_lengths=verification_lengths, return_state=true)
            state = if multi
                swe2d_topography_frames(verification_grid...; common..., mpi_dims=dims,
                                        print_error_metrics=false)
            else
                swe2d_topography_frames(; common..., nx_aoi=verification_grid[1],
                                        ny_aoi=verification_grid[2], sponge=false, perf_test=true)
            end
            if state !== nothing
                xs = range(-1, 1; length=verification_grid[1])[2:end-1]
                ys = range(-1, 1; length=verification_grid[2])[2:end-1]
                h0 = [max(0.0, case.surface(x, y) - case.bed(x, y)) for x in xs, y in ys]
                @test all(isfinite, state.h) && all(isfinite, state.hu) && all(isfinite, state.hv)
                @test minimum(state.h) >= 0.0
                if case.steady
                    @test maximum(abs.(state.h .- h0)) < 5e-14
                    @test maximum(abs.(state.hu)) < 5e-14
                    @test maximum(abs.(state.hv)) < 5e-14
                    @test all(iszero, state.h[h0 .== 0.0])
                    @test abs(sum(state.h) - sum(h0)) < 1e-12
                else
                    @test maximum(abs.(state.h .- h0)) > 1e-4
                    @test abs(sum(state.h) - sum(h0)) < 1e-12
                    if case.name == "dry_dam_break"
                        @test maximum(state.h[(h0 .== 0.0) .& (state.z .== 0.0)]) > 0.01
                    end
                end
                results[case.name] = state
                println(case.name, ": depth error=", maximum(abs.(state.h .- h0)),
                        ", max momentum=", max(maximum(abs.(state.hu)), maximum(abs.(state.hv))),
                        ", mass change=", sum(state.h) - sum(h0))
            end
        end
    end
    if !isempty(results)
        serialize(ARGS[2], results)
    end
    if multi
        MPI.Finalize()
    end
end

verify()
