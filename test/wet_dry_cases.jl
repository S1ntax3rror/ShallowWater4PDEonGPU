const verification_grid = (34, 30)
const verification_lengths = (2.0, 2.0)
const verification_steps = 200

function wet_dry_cases()
    island(x, y) = 0.6 * exp(-8 * (x^2 + 2 * y^2))
    wall(x, y) = max(abs(x), abs(y)) > 0.85 ? 1.0 : 0.0
    return [
        (name="fully_wet", bed=(x, y) -> 0.04 * sin(3*x) * cos(2*y), surface=(x, y) -> 0.2, steady=true),
        (name="island", bed=island, surface=(x, y) -> 0.2, steady=true),
        (name="oblique_shore", bed=(x, y) -> 0.6*x + 0.15*y, surface=(x, y) -> 0.2, steady=true),
        (name="bed_step", bed=(x, y) -> x + 0.3*y < 0.0 ? -0.1 : 0.5, surface=(x, y) -> 0.2, steady=true),
        (name="rough_shore", bed=(x, y) -> 0.2 + 0.3*sin(17*x)*cos(13*y), surface=(x, y) -> 0.2, steady=true),
        (name="all_dry", bed=(x, y) -> 0.5 + 0.1*x, surface=(x, y) -> 0.2, steady=true),
        (name="shallow_water", bed=(x, y) -> 0.0, surface=(x, y) -> 5e-12, steady=true),
        (name="dry_dam_break", bed=wall, surface=(x, y) -> x < -0.2 ? 0.25 : 0.0, steady=false),
        (name="shore_perturbation", bed=(x, y) -> max(island(x, y), wall(x, y)),
         surface=(x, y) -> 0.2 + 0.02*exp(-100*((x+0.5)^2+y^2)), steady=false),
    ]
end
