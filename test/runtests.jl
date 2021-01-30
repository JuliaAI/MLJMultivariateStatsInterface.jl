import Dates
import MLJMultivariateStatsInterface: _replace!
import MultivariateStats
import Random

using LinearAlgebra
using MLJBase
using MLJMultivariateStatsInterface
using StableRNGs
using Test

const MS = MultivariateStats

include("testutils.jl")
println("\nutils"); include("utils.jl")
println("\ncomponent_analysis"); include("models/decomposition_models.jl")
println("\ndiscriminant_analysis"); include("models/discriminant_analysis.jl")
println("\nlinear_models"); include("models/linear_models.jl")
