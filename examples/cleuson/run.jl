"""Run the timed Cleuson dam removal with the existing first-order XPU kernels."""
ENV["SWE_HEADLESS"] = "true"
include(joinpath(@__DIR__, "../../src/xpu/2d_swe_xpu_wb.jl"))
using TOML

@parallel_indices (ix, iy) function manning_friction!(h, hu, hv, gravity, n, dt)
    if h[ix, iy] > h_eps
        speed = sqrt(hu[ix, iy]^2 + hv[ix, iy]^2) / h[ix, iy]
        factor = 1 + dt * gravity * n^2 * speed / h[ix, iy]^(4 / 3)
        hu[ix, iy] /= factor
        hv[ix, iy] /= factor
    end
    return nothing
end

function run_case(case_dir, output_dir)
    cfg = TOML.parsefile(joinpath(case_dir, "case.toml"))
    nx, ny = cfg["nx"], cfg["ny"]
    dx, dy = cfg["dx"], cfg["dy"]
    gravity, n = cfg["gravity"], cfg["manning"]
    vel_eps = cfg["velocity_depth_scale"]^4
    breach_time, end_time = cfg["breach_time"], cfg["end_time"]
    interval, cfl = cfg["snapshot_interval"], cfg["cfl"]
    function read_field(name)
        a = zeros(nx, ny)
        read!(joinpath(case_dir, name * ".bin"), a)
        return a
    end
    h = read_field("h_initial")
    z = read_field("bed_before")
    z_after = read_field("bed_after")
    reservoir = read_field("reservoir_mask") .> 0
    initial_h = copy(h)
    hu, hv = zeros(nx, ny), zeros(nx, ny)
    sx, sy = zeros(nx - 1, ny), zeros(nx, ny - 1)
    F₁, F₂, F₃, Pₓ = (zeros(nx - 1, ny) for _ in 1:4)
    G₁, G₂, G₃, Pᵧ = (zeros(nx, ny - 1) for _ in 1:4)
    drain = zeros(nx, ny)
    tx, ty = zeros(nx - 1, ny), zeros(nx, ny - 1)
    interior = (2:nx-1, 2:ny-1)
    volume() = sum(@view h[interior...]) * dx * dy
    initial_volume = volume()
    outflow = 0.0
    min_raw_depth = 0.0
    max_balance_error = 0.0
    steady_depth_error = 0.0
    steady_momentum = 0.0
    time, step, frame = 0.0, 0, 0
    breached = false
    next_snapshot = interval
    maximum_depth = copy(h)
    arrival_time = fill(-1.0, nx, ny)
    mkpath(output_dir)
    log = open(joinpath(output_dir, "diagnostics.csv"), "w")
    println(log, "frame,time_s,volume_m3,reservoir_volume_m3,outflow_m3,balance_error_m3,max_depth_m,max_speed_m_s,min_raw_depth_m")
    function snapshot!()
        all(isfinite, h) && all(isfinite, hu) && all(isfinite, hv) || error("Nonfinite state at t=$time")
        minimum(h) >= 0 || error("Negative depth at t=$time")
        write(joinpath(output_dir, @sprintf("depth_%05d.bin", frame)), Float32.(h))
        wet = h .> 0.05
        speed = maximum(sqrt.(hu[wet].^2 .+ hv[wet].^2) ./ h[wet]; init=0.0)
        remaining = sum(h[reservoir]) * dx * dy
        balance = volume() + outflow - initial_volume
        @printf(log, "%d,%.9f,%.9f,%.9f,%.9f,%.9e,%.9f,%.9f,%.9e\n",
                frame, time, volume(), remaining, outflow, balance, maximum(h), speed, min_raw_depth)
        flush(log)
        frame += 1
    end
    snapshot!()
    wall_start = Base.time()
    while time < end_time - 1e-9
        step += 1
        step <= 200000 || error("Step limit exceeded at t=$time")
        @parallel compute_maxspeed!(sx, sy, h, hu, hv, z, gravity, vel_eps)
        rate = maximum(sx) / dx + maximum(sy) / dy
        event = breached ? end_time : breach_time
        dt = min(rate > 0 ? cfl / rate : interval, next_snapshot - time, event - time, end_time - time)
        isfinite(dt) && dt > 0 || error("Invalid timestep $dt at t=$time")
        @parallel compute_1st_2nd_and_3th_flux!(F₁, F₂, F₃, G₁, G₂, G₃, Pₓ, Pᵧ,
                                              hu, hv, h, z, gravity, sx, sy, vel_eps)
        @parallel compute_draining_timestep!(drain, F₁, G₁, h, dt, 1 / dx, 1 / dy)
        @parallel compute_effective_flux_timesteps!(tx, ty, drain, F₁, G₁, dt)
        # Integrate the same donor-limited boundary fluxes used in the update.
        outflow += dy * (sum(@view(F₁[nx-1, 2:ny-1]) .* @view(tx[nx-1, 2:ny-1])) -
                         sum(@view(F₁[1, 2:ny-1]) .* @view(tx[1, 2:ny-1]))) +
                   dx * (sum(@view(G₁[2:nx-1, ny-1]) .* @view(ty[2:nx-1, ny-1])) -
                         sum(@view(G₁[2:nx-1, 1]) .* @view(ty[2:nx-1, 1])))
        @parallel update_height_momentum!(h, hu, hv, F₁, G₁, F₂, F₃, G₂, G₃,
                                           tx, ty, Pₓ, Pᵧ, z, gravity, dt, 1 / dx, 1 / dy)
        min_raw_depth = min(min_raw_depth, minimum(@view h[interior...]))
        min_raw_depth >= -1e-9 || error("Positivity failure: $min_raw_depth")
        @parallel dry_cell_fix!(h, hu, hv, h_eps)
        @parallel manning_friction!(h, hu, hv, gravity, n, dt)
        @parallel (1:ny) left_right_bc!(h, hu, hv, z, gravity, dt, 1 / dx)
        @parallel (1:nx) bottom_top_bc!(h, hu, hv, z, gravity, dt, 1 / dy)
        @parallel dry_cell_fix!(h, hu, hv, h_eps)
        time += dt
        maximum_depth .= max.(maximum_depth, h)
        newly_wet = (arrival_time .< 0) .& (h .> 0.05) .& .!reservoir
        arrival_time[newly_wet] .= time
        max_balance_error = max(max_balance_error, abs(volume() + outflow - initial_volume))
        if !breached && time >= breach_time - 1e-9
            steady_depth_error = maximum(abs.(h .- initial_h))
            steady_momentum = max(maximum(abs.(hu)), maximum(abs.(hv)))
            steady_depth_error < 1e-8 && steady_momentum < 1e-7 || error("Initial reservoir is not at rest")
            mass_before = volume()
            z .= z_after
            volume() == mass_before || error("Dam removal changed the water volume")
            breached = true
            @printf("Dam removed at %.1f s; steady depth error %.3e m, momentum %.3e m²/s\n",
                    time, steady_depth_error, steady_momentum)
        end
        if time >= next_snapshot - 1e-9
            time = next_snapshot
            snapshot!()
            next_snapshot += interval
            if (frame - 1) % 40 == 0
                @printf("t=%6.1f s, step=%d, volume=%.3f Mm³, elapsed=%.1f s\n",
                        time, step, volume() / 1e6, Base.time() - wall_start)
                flush(stdout)
            end
        end
    end
    close(log)
    write(joinpath(output_dir, "maximum_depth.bin"), Float32.(maximum_depth))
    write(joinpath(output_dir, "arrival_time.bin"), Float32.(arrival_time))
    relative_balance = max_balance_error / initial_volume
    relative_balance < 1e-8 || error("Mass balance error $relative_balance")
    report = Dict("steps" => step, "frames" => frame, "initial_volume_m3" => initial_volume,
                  "final_volume_m3" => volume(), "net_outflow_m3" => outflow,
                  "max_mass_balance_error_m3" => max_balance_error,
                  "relative_mass_balance_error" => relative_balance,
                  "minimum_raw_depth_m" => min_raw_depth,
                  "steady_depth_error_m" => steady_depth_error,
                  "steady_momentum_m2_s" => steady_momentum,
                  "elapsed_seconds" => Base.time() - wall_start)
    open(joinpath(output_dir, "verification.toml"), "w") do io
        TOML.print(io, report)
    end
    for name in ("verification.toml", "diagnostics.csv")
        if abspath(case_dir) != abspath(output_dir)
            cp(joinpath(output_dir, name), joinpath(case_dir, name); force=true)
        end
    end
    @printf("Finished %d steps, %d frames; relative mass error %.3e\n", step, frame, relative_balance)
end

root = normpath(joinpath(@__DIR__, "../.."))
run_case(length(ARGS) >= 1 ? ARGS[1] : joinpath(root, "data/cleuson"),
         length(ARGS) >= 2 ? ARGS[2] : joinpath(root, "cache/swiss_dam/frames"))
