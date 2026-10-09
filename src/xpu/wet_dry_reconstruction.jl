# First-order hydrostatic reconstruction: both sides see the same bed barrier.
# Pressure corrections use these same depths, including at a blocked dry face.
@inline face_bed(zL, zR) = max(zL, zR)

@inline function boundary_depth(hB, zB, hI, zI, α)
    wB = hB + zB
    wI = hI + zI
    # A dry cell's bed elevation is not a water-surface elevation.
    if hB <= 0.0
        wB = min(zB, wI)
    end
    if hI <= 0.0
        wI = min(zI, wB)
    end
    return max(0.0, wI + α * (wI - wB) - zB)
end
