module MLJMultivariateStatsInterface

# ===================================================================
# IMPORTS
import MLJModelInterface
import MLJModelInterface: Table, Continuous, Finite, @mlj_model, metadata_pkg,
    metadata_model
import MultivariateStats
import MultivariateStats: SimpleCovariance
import StatsBase: CovarianceEstimator

using CategoricalDistributions
using Distances
using LinearAlgebra

# ===================================================================
## EXPORTS
# Models are exported automatically by `@mlj_model` macro

# ===================================================================
## Re-EXPORTS
export SimpleCovariance, CovarianceEstimator, SqEuclidean, CosineDist

# ===================================================================
## CONSTANTS
# Define constants for easy referencing of packages
const MMI = MLJModelInterface
const MS = MultivariateStats
const PCAFitResultType = MS.PCA
const KernelPCAFitResultType = MS.KernelPCA
const ICAFitResultType = MS.ICA
const PPCAFitResultType = MS.PPCA
const FactorAnalysisResultType = MS.FactorAnalysis
const default_kernel = (x, y) -> x'y #default kernel used in KernelPCA

# Definitions of model descriptions for use in model doc-strings.
const PKG = "MLJMultivariateStatsInterface"

# ===================================================================
# Includes
include("models/decomposition_models.jl")
include("models/discriminant_analysis.jl")
include("models/linear_models.jl")
include("utils.jl")

# ===================================================================
# List of all models interfaced
const MODELS = (
    LinearRegressor,
    MultitargetLinearRegressor,
    RidgeRegressor,
    MultitargetRidgeRegressor,
    PCA,
    KernelPCA,
    ICA,
    LDA,
    BayesianLDA,
    SubspaceLDA,
    BayesianSubspaceLDA,
    FactorAnalysis,
    PPCA
)

# ====================================================================
# PKG_METADATA
metadata_pkg.(
    MODELS,
    name = "MultivariateStats",
    uuid = "6f286f6a-111f-5878-ab1e-185364afe411",
    url = "https://github.com/JuliaStats/MultivariateStats.jl",
    license = "MIT",
    julia = true,
    is_wrapper = false
)

end
