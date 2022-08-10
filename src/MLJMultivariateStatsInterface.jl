module MLJMultivariateStatsInterface

# ===================================================================
# IMPORTS
import MLJModelInterface
import MLJModelInterface: Table, Continuous, Finite, @mlj_model, metadata_pkg,
    metadata_model
import MultivariateStats
import MultivariateStats: SimpleCovariance
import StatsBase: CovarianceEstimator

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

"""

$(MMI.doc_header(LinearRegressor))

`LinearRegressor` assumes the target is a `Continuous` variable and trains a linear
prediction function using the least squares algorithm. Options exist to specify a bias term.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
are of scitype `Continuous`; check the column scitypes with `schema(X)`.

- `y` is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new
  features `Xnew`, which should have the same scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `coefficients`: The linear coefficients determined by the model.
- `intercept`: The intercept determined by the model.

# Examples

```
using MLJ

LinearRegressor = @load LinearRegressor pkg=MultivariateStats
linear_regressor = LinearRegressor()

X, y = make_regression(100, 2) # a table and a vector (synthetic data)
mach = machine(linear_regressor, X, y) |> fit!

Xnew, _ = make_regression(3, 2)
yhat = predict(mach, Xnew) # new predictions
```

See also [`MultitargetLinearRegressor`](@ref), [`RidgeRegressor`](@ref),
[`MultitargetRidgeRegressor`](@ref)

"""
LinearRegressor

"""

$(MMI.doc_header(MultitargetLinearRegressor))

`MultitargetLinearRegressor` assumes the target variable is vector-valued with
continuous components.  It trains a linear prediction function using the
least squares algorithm. Options exist to specify a bias term.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
are of scitype `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any table of responses whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new
  features `Xnew`, which should have the same scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `coefficients`: The linear coefficients determined by the model.

- `intercept`: The intercept determined by the model.

# Examples

```
using MLJ
using DataFrames

LinearRegressor = @load MultitargetLinearRegressor pkg=MultivariateStats
linear_regressor = LinearRegressor()

X, y = make_regression(100, 9; n_targets = 2) # a table and a table (synthetic data)

mach = machine(linear_regressor, X, y) |> fit!

Xnew, _ = make_regression(3, 9)
yhat = predict(mach, Xnew) # new predictions
```

See also [`LinearRegressor`](@ref), [`RidgeRegressor`](@ref),
[`MultitargetRidgeRegressor`](@ref)

"""
MultitargetLinearRegressor

"""

$(MMI.doc_header(RidgeRegressor))

`RidgeRegressor` adds a quadratic penalty term to least squares regression, for
regularization. Ridge regression is particularly useful in the case of multicollinearity.
Options exist to specify a bias term, and to adjust the strength of the penalty term.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
are of scitype `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `lambda=1.0`: Is the non-negative parameter for the
  regularization strength. If lambda is 0, ridge regression is equivalent
  to linear least squares regression, and as lambda approaches infinity,
  all the linear coefficients approach 0.

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new
  features `Xnew`, which should have the same scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `coefficients`: The linear coefficients determined by the model.

- `intercept`: The intercept determined by the model.

# Examples

```
using MLJ

RidgeRegressor = @load RidgeRegressor pkg=MultivariateStats
pipe = Standardizer() |> RidgeRegressor(lambda=10)

X, y = @load_boston

mach = machine(pipe, X, y) |> fit!
yhat = predict(mach, X)
training_error = l1(yhat, y) |> mean
```

See also [`LinearRegressor`](@ref), [`MultitargetLinearRegressor`](@ref),
[`MultitargetRidgeRegressor`](@ref)

"""
RidgeRegressor

"""

$(MMI.doc_header(MultitargetRidgeRegressor))

Multi-target ridge regression adds a quadratic penalty term to multi-target least squares
regression, for regularization. Ridge regression is particularly useful in the case of
multicollinearity. In this case, the output represents a response vector. Options exist to
specify a bias term, and to adjust the strength of the penalty term.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
are of scitype `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any table of responses whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `lambda=1.0`: Is the non-negative parameter for the
  regularization strength. If lambda is 0, ridge regression is equivalent
  to linear least squares regression, and as lambda approaches infinity,
  all the linear coefficients approach 0.

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new
  features `Xnew`, which should have the same scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `coefficients`: The linear coefficients determined by the model.

- `intercept`: The intercept determined by the model.

# Examples

```
using MLJ
using DataFrames

RidgeRegressor = @load MultitargetRidgeRegressor pkg=MultivariateStats

X, y = make_regression(100, 6; n_targets = 2)  # a table and a table (synthetic data)

ridge_regressor = RidgeRegressor(lambda=1.5)
mach = machine(ridge_regressor, X, y) |> fit!

Xnew, _ = make_regression(3, 6)
yhat = predict(mach, Xnew) # new predictions
```

See also [`LinearRegressor`](@ref), [`MultitargetLinearRegressor`](@ref),
[`RidgeRegressor`](@ref)

"""
MultitargetRidgeRegressor

"""

$(MMI.doc_header(PCA))

Principal component analysis learns a linear projection onto a lower dimensional space while
preserving most of the initial variance seen in the training data.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `maxoutdim=0`: Together with `variance_ratio`, controls the output dimension `outdim`
   chosen by the model. Specifically, suppose that `k` is the smallest integer such that
   retaining the `k` most significant principal components accounts for `variance_ratio` of
   the total variance in the training data. Then `outdim = min(outdim, maxoutdim)`. If
   `maxoutdim=0` (default) then the effective `maxoutdim` is `min(n, indim - 1)` where `n`
   is the number of observations and `indim` the number of features in the training data.

- `variance_ratio::Float64=0.99`: The ratio of variance preserved after the transformation

- `method=:auto`: The method to use to solve the problem. Choices are

    - `:svd`: Support Vector Decomposition of the matrix.

    - `:cov`: Covariance matrix decomposition.

    - `:auto`: Use `:cov` if the matrices first dimension is smaller than its second
      dimension and otherwise use `:svd`

- `mean=nothing`: if `nothing`, centering will be computed and applied, if set to `0` no
  centering (data is assumed pre-centered); if a vector is passed, the centering is done
  with that vector.

# Operations

- `transform(mach, Xnew)`: Return a lower dimensional projection of the input `Xnew`, which
  should have the same scitype as `X` above.

- `inverse_transform(mach, Xsmall)`: For a dimension-reduced table `Xsmall`,
  such as returned by `transform`, reconstruct a table, having same the number
  of columns as the original training data `X`, that transforms to `Xsmall`.
  Mathematically, `inverse_transform` is a right-inverse for the PCA projection
  map, whose image is orthogonal to the kernel of that map. In particular, if
  `Xsmall = transform(mach, Xnew)`, then `inverse_transform(Xsmall)` is
  only an approximation to `Xnew`.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projection`: Returns the projection matrix, which has size `(indim, outdim)`, where
  `indim` and `outdim` are the number of features of the input and output respectively.

# Report

The fields of `report(mach)` are:

- `indim`: Dimension (number of columns) of the training data and new data to be transformed.

- `outdim = min(n, indim, maxoutdim)` is the output dimension; here `n` is the number of
  observations.

- `tprincipalvar`: Total variance of the principal components.

- `tresidualvar`: Total residual variance.

- `tvar`: Total observation variance (principal + residual variance).

- `mean`: The mean of the untransformed training data, of length `indim`.

- `principalvars`: The variance of the principal components.

# Examples

```
using MLJ

PCA = @load PCA pkg=MultivariateStats

X, y = @load_iris # a table and a vector

model = PCA(maxoutdim=2)
mach = machine(model, X) |> fit!

Xproj = transform(mach, X)
```

See also [`KernelPCA`](@ref), [`ICA`](@ref), [`FactorAnalysis`](@ref), [`PPCA`](@ref)

"""
PCA

"""

$(MMI.doc_header(KernelPCA))

In kernel PCA the linear operations of ordinary principal component analysis are performed
in a [reproducing Hilbert
space](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space).

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `maxoutdim=0`: Controls the the dimension (number of columns) of the output,
  `outdim`. Specifically, `outdim = min(n, indim, maxoutdim)`, where `n` is the number of
  observations and `indim` the input dimension.

- `kernel::Function=(x,y)->x'y`: The kernel function, takes in 2 vector arguments
  x and y, returns a scalar value. Defaults to the dot product of `x` and `y`.

- `solver::Symbol=:eig`: solver to use for the eigenvalues, one of `:eig`(default, uses
  `LinearAlgebra.eigen`), `:eigs`(uses `Arpack.eigs`).

- `inverse::Bool=true`: perform calculations needed for inverse transform

- `beta::Real=1.0`: strength of the ridge regression that learns the inverse transform
  when inverse is true.

- `tol::Real=0.0`: Convergence tolerance for eigenvalue solver.

- `maxiter::Int=300`: maximum number of iterations for eigenvalue solver.

# Operations

- `transform(mach, Xnew)`: Return a lower dimensional projection of the input `Xnew`, which
  should have the same scitype as `X` above.

- `inverse_transform(mach, Xsmall)`: For a dimension-reduced table `Xsmall`, such as
  returned by `transform`, reconstruct a table, having same the number of columns as the
  original training data `X`, that transforms to `Xsmall`.  Mathematically,
  `inverse_transform` is a right-inverse for the PCA projection map, whose image is
  orthogonal to the kernel of that map. In particular, if `Xsmall = transform(mach, Xnew)`,
  then `inverse_transform(Xsmall)` is only an approximation to `Xnew`.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projection`: Returns the projection matrix, which has size `(indim, outdim)`, where
  `indim` and `outdim` are the number of features of the input and ouput respectively.

# Report

The fields of `report(mach)` are:

- `indim`: Dimension (number of columns) of the training data and new data to be
  transformed.

- `outdim`: Dimension of transformed data.

- `principalvars`: The variance of the principal components.

# Examples

```
using MLJ
using LinearAlgebra

KernelPCA = @load KernelPCA pkg=MultivariateStats

X, y = @load_iris # a table and a vector

function rbf_kernel(length_scale)
    return (x,y) -> norm(x-y)^2 / ((2 * length_scale)^2)
end

model = KernelPCA(maxoutdim=2, kernel=rbf_kernel(1))
mach = machine(model, X) |> fit!

Xproj = transform(mach, X)
```

See also
[`PCA`](@ref), [`ICA`](@ref), [`FactorAnalysis`](@ref), [`PPCA`](@ref)
"""
KernelPCA

"""
$(MMI.doc_header(ICA))

Independent component analysis is a computational technique for separating a multivariate
signal into additive subcomponents, with the assumption that the subcomponents are
non-Gaussian and independent from each other.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `outdim::Int=0`: The number of independent components to recover, set automatically if `0`.

- `alg::Symbol=:fastica`: The algorithm to use (only `:fastica` is supported at the moment).

- `fun::Symbol=:tanh`: The approximate neg-entropy function, one of `:tanh`, `:gaus`.

- `do_whiten::Bool=true`: Whether or not to perform pre-whitening.

- `maxiter::Int=100`: The maximum number of iterations.

- `tol::Real=1e-6`: The convergence tolerance for change in the unmixing matrix W.

- `mean::Union{Nothing, Real, Vector{Float64}}=nothing`: mean to use, if nothing (default)
  centering is computed and applied, if zero, no centering; otherwise a vector of means can
  be passed.

- `winit::Union{Nothing,Matrix{<:Real}}=nothing`: Initial guess for the unmixing matrix `W`:
  either an empty matrix (for random initialization of `W`), a matrix of size `m × k` (if
  `do_whiten` is true), or a matrix of size `m × k`. Here `m` is the number of components
  (columns) of the input.

# Operations

- `transform(mach, Xnew)`: Return the component-separated version of input
  `Xnew`, which should have the same scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projection`: The estimated component matrix.

- `mean`: The estimated mean vector.

# Report

The fields of `report(mach)` are:

- `indim`: Dimension (number of columns) of the training data and new data to be transformed.

- `outdim`: Dimension of transformed data.

- `mean`: The mean of the untransformed training data, of length `indim`.

# Examples

```
using MLJ

ICA = @load ICA pkg=MultivariateStats

times = range(0, 8, length=2000)

sine_wave = sin.(2*times)
square_wave = sign.(sin.(3*times))
sawtooth_wave = map(t -> mod(2t, 2) - 1, times)
signals = hcat(sine_wave, square_wave, sawtooth_wave)
noisy_signals = signals + 0.2*randn(size(signals))

mixing_matrix = [ 1 1 1; 0.5 2 1; 1.5 1 2]
X = MLJ.table(noisy_signals*mixing_matrix)

model = ICA(outdim = 3, tol=0.1)
mach = machine(model, X) |> fit!

X_unmixed = transform(mach, X)

using Plots

plot(X.x2)
plot(X.x2)
plot(X.x3)

plot(X_unmixed.x1)
plot(X_unmixed.x2)
plot(X_unmixed.x3)

```

See also
[`PCA`](@ref), [`KernelPCA`](@ref), [`FactorAnalysis`](@ref), [`PPCA`](@ref)

"""
ICA

"""
$(MMI.doc_header(LDA))

[Multiclass linear discriminant
analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) learns a projection in
a space of features to a lower dimensional space, in a way that attempts to preserve as much
as possible the degree to which the classes of a discrete target variable can be
discriminated. This can be used either for dimension reduction of the features (see
`transform` below) or for probabilistic classification of the target (see `predict` below).

In the case of prediction, the class probability for a new observation reflects the
proximity of that observation to training observations associated with that class, and how
far away the observation is from observations associated with other classes. Specifically,
the distances, in the transformed (projected) space, of a new observation, from the centroid
of each target class, is computed; the resulting vector of distances, multiplied by minus
one, is passed to a softmax function to obtain a class probability prediction. Here
"distance" is computed using a user-specified distance function.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
are of scitype `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any `AbstractVector` whose element
  scitype is `OrderedFactor` or `Multiclass`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `method::Symbol=:gevd`: The solver, one of `:gevd` or `:whiten` methods.

- `cov_w::StatsBase.SimpleCovariance()`: An estimator for the within-class
  covariance (used in computing the within-class scatter matrix, `Sw`). Any robust estimator
  from `CovarianceEstimation.jl` can be used.

- `cov_b::StatsBase.SimpleCovariance()`: The same as `cov_w` but for the
  between-class covariance (used in computing the between-class scatter matrix, `Sb`).

- `outdim::Int=0`: The output dimension, i.e dimension of the transformed space,
  automatically set if equal to 0.

- `regcoef::Float64=1e-6`: The regularization coefficient. A positive value
  `regcoef*eigmax(Sw)` where `Sw` is the within-class scatter matrix, is added to the
  diagonal of `Sw` to improve numerical stability. This can be useful if using the standard
  covariance estimator.

- `dist=Distances.SqEuclidean()`: The distance metric to use when performing classification
  (to compare the distance between a new point and centroids in the transformed space); must
  be a subtype of `Distances.SemiMetric` from Distances.jl, e.g., `Distances.CosineDist`.

# Operations

- `transform(mach, Xnew)`: Return a lower dimensional projection of the input `Xnew`, which
  should have the same scitype as `X` above.

- `predict(mach, Xnew)`: Return predictions of the target given features `Xnew` having the
  same scitype as `X` above. Predictions are probabilistic but uncalibrated.

- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions returned
  above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projected_class_means`: The matrix comprised of class-specific means as columns, of size
  `(indim, nclasses)`, where `indim` is the number of input features (columns) and
  `nclasses` the number of target classes.

- `projection_matrix`: The learned projection matrix, of size `(indim, outdim)`, where
 `indim` and `outdim` are the input and output dimensions respectively.

# Report

The fields of `report(mach)` are:

- `classes`: The classes seen during model fitting.

- `outdim`: The dimensions the model is projected to.

- `projected_class_means`: The matrix comprised of class-specific means as
  columns (see above).

- `mean`: The mean of the untransformed training data, of length `indim`.

- `class_weights`: The weights of each class.

- `Sb`: The between class scatter matrix.

- `Sw`: The within class scatter matrix.

- `nclasses`: The number of classes directly observed in the training data (which can be
  less than the total number of classes in the class pool)

# Examples

```
using MLJ

LDA = @load LDA pkg=MultivariateStats

X, y = @load_iris # a table and a vector

model = LDA()
mach = machine(model, X, y) |> fit!

Xproj = transform(mach, X)
y_hat = predict(mach, X)
labels = predict_mode(mach, X)

```

See also [`BayesianLDA`](@ref), [`SubspaceLDA`](@ref), [`BayesianSubspaceLDA`](@ref)

"""
LDA

"""

$(MMI.doc_header(BayesianLDA))

The Bayesian multiclass LDA algorithm learns a projection matrix as described in ordinary
[`LDA`](@ref).  Predicted class posterior probability distributions are derived by applying
Bayes' rule with a multivariate Gaussian class-conditional distribution. A prior class
distribution can be specified by the user or inferred from training data class frequency.

See also the [package
documentation](https://multivariatestatsjl.readthedocs.io/en/latest/lda.html).  For more
information about the algorithm, see [Li, Zhu and Ogihara (2006): Using Discriminant
Analysis for Multi-class Classification: An Experimental
Investigation](https://doi.org/10.1007/s10115-006-0013-y).

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any `AbstractVector` whose element scitype is
  `OrderedFactor` or `Multiclass`; check the scitype with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `method::Symbol=:gevd`: choice of solver, one of `:gevd` or `:whiten` methods.

- `cov_w::StatsBase.SimpleCovariance()`: An estimator for the within-class
  covariance (used in computing the within-class scatter matrix, `Sw`). Any robust estimator
  from `CovarianceEstimation.jl` can be used.

- `cov_b::StatsBase.SimpleCovariance()`: The same as `cov_w` but for the
  between-class covariance (used in computing the between-class scatter matrix, `Sb`).

- `outdim::Int=0`: The output dimension, i.e., dimension of the transformed space,
  automatically set if equal to 0.

- `regcoef::Float64=1e-6`: The regularization coefficient. A positive value
  `regcoef*eigmax(Sw)` where `Sw` is the within-class scatter matrix, is added to the
  diagonal of `Sw` to improve numerical stability. This can be useful if using the standard
  covariance estimator.

- `priors::Union{Nothing, Vector{Float64}}=nothing`: For use in prediction with Bayes'
  rule. If `priors = nothing` then `priors` are estimated from the class proportions in the
  training data. Otherwise it requires a `Vector` containing class probabilities with
  probabilities specified using the order given by `levels(y)`, where `y` is the training
  target.

# Operations

- `transform(mach, Xnew)`: Return a lower dimensional projection of the input `Xnew`, which
  should have the same scitype as `X` above.

- `predict(mach, Xnew)`: Return predictions of the target given features `Xnew`, which
  should have the same scitype as `X` above. Predictions are probabilistic but uncalibrated.

- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions returned
  above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projected_class_means`: The matrix comprised of class-specific means as columns, of size
  `(indim, nclasses)`, where `indim` is the number of input features (columns) and
  `nclasses` the number of target classes.

- `projection_matrix`: The learned projection matrix, of size `(indim, outdim)`, where
 `indim` and `outdim` are the input and output dimensions respectively.

- `priors`: The class priors for classification. As inferred from training target `y`, if
  not user-specified. A vector with order consistent with `levels(y)`.

# Report

The fields of `report(mach)` are:

- `classes`: The classes seen during model fitting.

- `outdim`: The dimensions the model is projected to.

- `projected_class_means`: The matrix comprised of class-specific means as columns (see
  above).

- `mean`: The mean of the untransformed training data, of length `indim`.

- `class_weights`: The weights of each class.

- `Sb`: The between class scatter matrix.

- `Sw`: The within class scatter matrix.

- `nclasses`: The number of classes directly observed in the training data (which can be
  less than the total number of classes in the class pool)

# Examples

```
using MLJ

BayesianLDA = @load BayesianLDA pkg=MultivariateStats

X, y = @load_iris # a table and a vector

model = BayesianLDA()
mach = machine(model, X, y) |> fit!

Xproj = transform(mach, X)
y_hat = predict(mach, X)
labels = predict_mode(mach, X)
```

See also [`LDA`](@ref), [`SubspaceLDA`](@ref), [`BayesianSubspaceLDA`](@ref)

"""
BayesianLDA

"""

$(MMI.doc_header(SubspaceLDA))

Multiclass subspace linear discriminant analysis (LDA) is a variation on ordinary
[`LDA`](@ref) suitable for high dimensional data, as it avoids storing scatter matrices. For
details, refer the [MultivariateStats.jl
documentation](https://juliastats.org/MultivariateStats.jl/stable/).

In addition to dimension reduction (using `transform`) probabilistic classification is
provided (using `predict`).  In the case of classification, the class probability for a new
observation reflects the proximity of that observation to training observations associated
with that class, and how far away the observation is from observations associated with other
classes. Specifically, the distances, in the transformed (projected) space, of a new
observation, from the centroid of each target class, is computed; the resulting vector of
distances, multiplied by minus one, is passed to a softmax function to obtain a class
probability prediction. Here "distance" is computed using a user-specified distance
function.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
are of scitype `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any `AbstractVector` whose element
  scitype is `OrderedFactor` or `Multiclass`; check the scitype
  with `scitype(y)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `normalize=true`: Option to normalize the between class variance for the number of
  observations in each class, one of `true` or `false`.

- `outdim`: the ouput dimension, automatically set if equal to `0`. If a non-zero `outdim`
  is passed, then the actual output dimension used is `min(rank, outdim)` where `rank` is
  the rank of the within-class covariance matrix.

- `dist=Distances.SqEuclidean()`: The distance metric to use when performing classification
  (to compare the distance between a new point and centroids in the transformed space); must
  be a subtype of `Distances.SemiMetric` from Distances.jl, e.g., `Distances.CosineDist`.


# Operations

- `transform(mach, Xnew)`: Return a lower dimensional projection of the input `Xnew`, which
  should have the same scitype as `X` above.

- `predict(mach, Xnew)`: Return predictions of the target given features `Xnew`, which
  should have same scitype as `X` above. Predictions are probabilistic but uncalibrated.

- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions
  returned above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projected_class_means`: The matrix comprised of class-specific means as columns, of size
  `(indim, nclasses)`, where `indim` is the number of input features (columns) and
  `nclasses` the number of target classes.

- `projection_matrix`: The learned projection matrix, of size `(indim, outdim)`, where
  `indim` and `outdim` are the input and output dimensions respectively.

# Report

The fields of `report(mach)` are:

- `explained_variance_ratio`: The ratio of explained variance to total variance. Each
  dimension corresponds to an eigenvalue.

- `classes`: The classes seen during model fitting.

- `projected_class_means`: The matrix comprised of class-specific means as columns (see
  above).

- `mean`: The mean of the untransformed training data, of length `indim`.

- `class_weights`: The weights of each class.

- `nclasses`: The number of classes directly observed in the training data (which can be
  less than the total number of classes in the class pool)

# Examples

```
using MLJ

SubspaceLDA = @load SubspaceLDA pkg=MultivariateStats

X, y = @load_iris # a table and a vector

model = SubspaceLDA()
mach = machine(model, X, y) |> fit!

Xproj = transform(mach, X)
y_hat = predict(mach, X)
labels = predict_mode(mach, X)
```

See also [`LDA`](@ref), [`BayesianLDA`](@ref), [`BayesianSubspaceLDA`](@ref)

"""
SubspaceLDA

"""
$(MMI.doc_header(BayesianSubspaceLDA))

The Bayesian multiclass subspace linear discriminant analysis algorithm learns a projection
matrix as described in [`SubspaceLDA`](@ref). The posterior class probability distribution
is derived as in [`BayesianLDA`](@ref).


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any `AbstractVector` whose element scitype is
  `OrderedFactor` or `Multiclass`; check the scitype with `scitype(y)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `normalize=true`: Option to normalize the between class variance for the number of
  observations in each class, one of `true` or `false`.

- `outdim`: the ouput dimension, automatically set if equal to `0`. If a non-zero `outdim`
  is passed, then the actual output dimension used is `min(rank, outdim)` where `rank` is
  the rank of the within-class covariance matrix.

- `priors::Union{Nothing, Vector{Float64}}=nothing`: For use in prediction with Bayes'
  rule. If `priors = nothing` then `priors` are estimated from the class proportions
  in the training data. Otherwise it requires a `Vector` containing class
  probabilities with probabilities specified using the order given by `levels(y)`
  where `y` is the training target.


# Operations

- `transform(mach, Xnew)`: Return a lower dimensional projection of the input `Xnew`, which
  should have the same scitype as `X` above.

- `predict(mach, Xnew)`: Return predictions of the target given features `Xnew`, which
  should have same scitype as `X` above. Predictions are probabilistic but uncalibrated.

- `predict_mode(mach, Xnew)`: Return the modes of the probabilistic predictions
  returned above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projected_class_means`: The matrix comprised of class-specific means as columns,
  of size `(indim, nclasses)`, where `indim` is the number of input features (columns) and
  `nclasses` the number of target classes.

- `projection_matrix`: The learned projection matrix, of size `(indim, outdim)`, where
 `indim` and `outdim` are the input and output dimensions respectively.

- `priors`: The class priors for classification. As inferred from training target `y`,
  if not user-specified. A vector with order consistent with `levels(y)`.

# Report

The fields of `report(mach)` are:

- `explained_variance_ratio`: The ratio of explained variance to total variance. Each
  dimension corresponds to an eigenvalue.

- `classes`: The classes seen during model fitting.

- `projected_class_means`: The matrix comprised of class-specific means as columns (see
  above).

- `mean`: The mean of the untransformed training data, of length `indim`.

- `class_weights`: The weights of each class.

- `nclasses`: The number of classes directly observed in the training data (which can be
  less than the total number of classes in the class pool)

# Examples

```
using MLJ

BayesianSubspaceLDA = @load BayesianSubspaceLDA pkg=MultivariateStats

X, y = @load_iris # a table and a vector

model = BayesianSubspaceLDA()
mach = machine(model, X, y) |> fit!

Xproj = transform(mach, X)
y_hat = predict(mach, X)
labels = predict_mode(mach, X)
```

See also [`LDA`](@ref), [`BayesianLDA`](@ref), [`SubspaceLDA`](@ref)

"""
BayesianSubspaceLDA

"""

$(MMI.doc_header(FactorAnalysis))

Factor analysis is a linear-Gaussian latent variable model that is closely related to
probabilistic PCA. In contrast to the probabilistic PCA model, the covariance of conditional
distribution of the observed variable given the latent variable is diagonal rather than
isotropic.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `method::Symbol=:cm`: Method to use to solve the problem, one of `:ml`, `:em`, `:bayes`.

- `maxoutdim=0`: Controls the the dimension (number of columns) of the output,
  `outdim`. Specifically, `outdim = min(n, indim, maxoutdim)`, where `n` is the number of
  observations and `indim` the input dimension.

- `maxiter::Int=1000`: Maximum number of iterations.

- `tol::Real=1e-6`: Convergence tolerance.

- `eta::Real=tol`: Variance lower bound.

- `mean::Union{Nothing, Real, Vector{Float64}}=nothing`: If `nothing`, centering will be
  computed and applied; if set to `0` no centering is applied (data is assumed
  pre-centered); if a vector, the centering is done with that vector.

# Operations

- `transform(mach, Xnew)`: Return a lower dimensional projection of the input `Xnew`, which
  should have the same scitype as `X` above.

- `inverse_transform(mach, Xsmall)`: For a dimension-reduced table `Xsmall`,
  such as returned by `transform`, reconstruct a table, having same the number
  of columns as the original training data `X`, that transforms to `Xsmall`.
  Mathematically, `inverse_transform` is a right-inverse for the PCA projection
  map, whose image is orthogonal to the kernel of that map. In particular, if
  `Xsmall = transform(mach, Xnew)`, then `inverse_transform(Xsmall)` is
  only an approximation to `Xnew`.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projection`: Returns the projection matrix, which has size `(indim, outdim)`, where
  `indim` and `outdim` are the number of features of the input and ouput respectively.
  Each column of the projection matrix corresponds to a factor.

# Report

The fields of `report(mach)` are:

- `indim`: Dimension (number of columns) of the training data and new data to be transformed.

- `outdim`: Dimension of transformed data (number of factors).

- `variance`: The variance of the factors.

- `covariance_matrix`: The estimated covariance matrix.

- `mean`: The mean of the untransformed training data, of length `indim`.

- `loadings`: The factor loadings.

# Examples

```
using MLJ

FactorAnalysis = @load FactorAnalysis pkg=MultivariateStats

X, y = @load_iris # a table and a vector

model = FactorAnalysis(maxoutdim=2)
mach = machine(model, X) |> fit!

Xproj = transform(mach, X)
```

See also [`KernelPCA`](@ref), [`ICA`](@ref), [`PPCA`](@ref), [`PCA`](@ref)

"""
FactorAnalysis

"""

$(MMI.doc_header(PPCA))

Probabilistic principal component analysis is a dimension-reduction algorithm which
represents a constrained form of the Gaussian distribution in which the number of free
parameters can be restricted while still allowing the model to capture the dominant
correlations in a data set. It is expressed as the maximum likelihood solution of a
probabilistic latent variable model. For details, see Bishop (2006): C. M. Pattern
Recognition and Machine Learning.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
  `Continuous`; check column scitypes with `schema(X)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `maxoutdim=0`: Controls the the dimension (number of columns) of the output,
  `outdim`. Specifically, `outdim = min(n, indim, maxoutdim)`, where `n` is the number of
  observations and `indim` the input dimension.

- `method::Symbol=:ml`: The method to use to solve the problem, one of `:ml`, `:em`,
  `:bayes`.

- `maxiter::Int=1000`: The maximum number of iterations.

- `tol::Real=1e-6`: The convergence tolerance.

- `mean::Union{Nothing, Real, Vector{Float64}}=nothing`: If `nothing`, centering will be
  computed and applied; if set to `0` no centering is applied (data is assumed
  pre-centered); if a vector, the centering is done with that vector.

# Operations

- `transform(mach, Xnew)`: Return a lower dimensional projection of the input `Xnew`, which
  should have the same scitype as `X` above.

- `inverse_transform(mach, Xsmall)`: For a dimension-reduced table `Xsmall`,
  such as returned by `transform`, reconstruct a table, having same the number
  of columns as the original training data `X`, that transforms to `Xsmall`.
  Mathematically, `inverse_transform` is a right-inverse for the PCA projection
  map, whose image is orthogonal to the kernel of that map. In particular, if
  `Xsmall = transform(mach, Xnew)`, then `inverse_transform(Xsmall)` is
  only an approximation to `Xnew`.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projection`: Returns the projection matrix, which has size `(indim, outdim)`, where
  `indim` and `outdim` are the number of features of the input and ouput respectively.
  Each column of the projection matrix corresponds to a principal component.

# Report

The fields of `report(mach)` are:

- `indim`: Dimension (number of columns) of the training data and new data to be transformed.
- `outdim`: Dimension of transformed data.

- `tvat`: The variance of the components.

- `loadings`: The models loadings, weights for each variable used when calculating principal
  components.

# Examples

```
using MLJ

PPCA = @load PPCA pkg=MultivariateStats

X, y = @load_iris # a table and a vector

model = PPCA(maxoutdim=2)
mach = machine(model, X) |> fit!

Xproj = transform(mach, X)
```

See also
[`KernelPCA`](@ref), [`ICA`](@ref), [`FactorAnalysis`](@ref), [`PCA`](@ref)
"""
PPCA

end
