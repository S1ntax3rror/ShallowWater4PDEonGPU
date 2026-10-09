using Test
using Serialization
using MPI

function runtests()
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    project = dirname(Base.active_project())
    driver = joinpath(@__DIR__, "wet_dry_driver.jl")
    mktempdir() do output
        single_file = joinpath(output, "single.jls")
        run(`$julia --project=$project --startup-file=no -t 2 $driver single $single_file`)
        single = deserialize(single_file)
        for dims in [(1, 1), (2, 1), (2, 2)]
            result_file = joinpath(output, "multi_$(dims[1])_$(dims[2]).jls")
            run(`$(MPI.mpiexec()) -n $(prod(dims)) $julia --project=$project --startup-file=no -t 2 $driver multi $result_file $(dims[1]) $(dims[2])`)
            distributed = deserialize(result_file)
            @testset "XPU vs MPI $(dims)" begin
                max_difference = 0.0
                for name in keys(single)
                    for field in (:h, :hu, :hv, :z)
                        difference = maximum(abs.(getproperty(single[name], field) .-
                                                   getproperty(distributed[name], field)))
                        max_difference = max(max_difference, difference)
                        @test difference < 1e-13
                    end
                    @test isapprox(single[name].time, distributed[name].time; atol=1e-13, rtol=1e-13)
                end
                println("Maximum XPU/MPI state difference for ", dims, ": ", max_difference)
            end
        end
    end
end

runtests()
