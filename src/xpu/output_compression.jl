using Serialization
using Tar
using CodecZlib
using Printf

# Define our maximum dimension for each "Quality" tag
const QUALITIES = Dict(
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

    # Read the first file to determine original dimensions and calculate scaling
    first_file = joinpath(indir, files[1])
    data = deserialize(first_file)
    nx, ny = size(data.h)
    
    # Calculate new dimensions while maintaining the aspect ratio
    scale = min(1.0, max_dim / max(nx, ny))
    new_nx = max(1, round(Int, nx * scale))
    new_ny = max(1, round(Int, ny * scale))
    
    println("Original size: $(nx)x$(ny) -> New downsampled size: $(new_nx)x$(new_ny)")

    # Downsample all arrays
    for (i, f) in enumerate(files)
        inpath = joinpath(indir, f)
        outpath = joinpath(outdir, f)
        
        frame_data = deserialize(inpath)
        
        h_small = downsample_array(frame_data.h, new_nx, new_ny)

        if i == 1
            z_small = downsample_array(frame_data.z, new_nx, new_ny)
        
            # Serialize back to the temporary folder
            serialize(outpath, (h=h_small, z=z_small))
        else
            # Serialize only the downsampled h for subsequent frames (z is static)
            serialize(outpath, (h=h_small,))
        end
        
        if i % 5 == 0 || i == length(files)
            percent = 100 * i / length(files)
            @printf("\rDownsampling: %.1f%%", percent)
            flush(stdout)
        end
    end
    println("\nDownsampling complete.")

    # Package into a tar.gz
    tarball = joinpath(indir, "compressed_frames_$(quality).tar.gz")
    println("Creating cross-file compressed archive (this might take a minute)...")
    
    # level=9 for maximum compression
    open(tarball, "w") do file
        stream = GzipCompressorStream(file; level=9)
        Tar.create(outdir, stream)
        close(stream)
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