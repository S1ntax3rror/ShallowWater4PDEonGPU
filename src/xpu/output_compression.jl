using Serialization
using Tar
using CodecZlib
using Printf
using Base.Threads
using CUDA

# Define our maximum dimension for each "Quality" tag
const QUALITIES = Dict(
    "TEST" => 50,
    "SD"  => 512,
    "HD"  => 1024,
    "FHD" => 1920,
    "QHD" => 2560,
    "4K"  => 3840
)

# Nearest-neighbor downsampling for 2D arrays
function downsample_array(arr, new_nx, new_ny)
    nx, ny = size(arr)
    # Generate scaled indices
    idx_x = round.(Int, range(1, nx, length=new_nx))
    idx_y = round.(Int, range(1, ny, length=new_ny))
    return arr[idx_x, idx_y]
end

function process_folder(quality::String, indir::String)
    quality = uppercase(quality)
    if !haskey(QUALITIES, quality)
        println("Error: Unknown quality '$quality'. Available options: ", join(keys(QUALITIES), ", "))
        return
    end

    max_dim = QUALITIES[quality]
    
    if !isdir(indir)
        println("Error: Directory '$indir' not found.")
        return
    end

    # Create a temporary folder for the downsampled frames
    outdir = joinpath(indir, "temp_downsampled_$(quality)")
    mkpath(outdir)

    # Find all saved array files
    files = filter(f -> endswith(f, ".jls"), readdir(indir))
    sort!(files) # Ensure chronological order

    if isempty(files)
        println("No .jls files found in $indir.")
        return
    end

    println("Found $(length(files)) frames. Compressing to $quality (max dimension $max_dim)...")

    # Determine orig dimensions and calc scaling
    first_file = joinpath(indir, files[1])
    data = deserialize(first_file)
    nx, ny = size(data.h)
    
    # Calc new dimensions while maintaining the aspect ratio
    scale = min(1.0, max_dim / max(nx, ny))
    new_nx = max(1, round(Int, nx * scale))
    new_ny = max(1, round(Int, ny * scale))
    
    println("Original size: $(nx)x$(ny) -> New downsampled size: $(new_nx)x$(new_ny)")
    println("Using $(Threads.nthreads()) threads for processing...")

    completed = Threads.Atomic{Int}(0)
    total_files = length(files)

    # Downsample all arrays
    Threads.@threads for i in eachindex(files)
        f = files[i]

        inpath = joinpath(indir, f)
        outpath = joinpath(outdir, f)
        
        frame_data = deserialize(inpath)
        h_small = downsample_array(frame_data.h, new_nx, new_ny)

        if haskey(frame_data, :z)
            z_small = downsample_array(frame_data.z, new_nx, new_ny)
            serialize(outpath, (h=h_small, z=z_small))
        else
            serialize(outpath, (h=h_small,))
        end

        Threads.atomic_add!(completed, 1)
        curr = completed[]

        if curr % max(1, div(total_files, 20)) == 0 || curr == total_files
            percent = 100 * curr / total_files
            print("\rDownsampling: $(round(percent, digits=1))% ($curr/$total_files)")
        end
    end
    println("\nDownsampling complete.")

    # Package into a tar.gz
    tarball = joinpath(indir, "compressed_frames_$(quality).tar.gz")
    println("Creating cross-file compressed archive (this might take a minute)...")
    
    try
        # Attempt to use 'pigz' (parallel gzip)
        n_cores = Threads.nthreads()
        run(pipeline(`tar -cf - -C $(dirname(outdir)) $(basename(outdir))`, `pigz -p $n_cores -9`, tarball))
        println("Archive created blazingly fast using system pigz!")
    catch e
        println("System 'pigz' not found. Falling back to single-threaded Julia compression (this may take a few minutes)...")
        open(tarball, "w") do file
            stream = GzipCompressorStream(file; level=9)
            Tar.create(outdir, stream)
            close(stream)
        end
    end
    
    println("Archive ready for download: $tarball")
    
    # Clean up the temporary unzipped files
    println("Cleaning up temporary directory...")
    rm(outdir, recursive=true)
    println("Done!")
end

# CLI Entry Point
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 2
        println("Usage: julia compress_frames.jl <QUALITY> <DIRECTORY>")
        println("Example: julia compress_frames.jl HD docs/frames/frames_topography")
    else
        process_folder(ARGS[1], ARGS[2])
    end
end