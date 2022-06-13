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
const PCA_DESCR = """
      Principal component analysis. Learns a linear transformation to
    project the data  on a lower dimensional space while preserving most of the initial
    variance.
    """
const KPCA_DESCR = "Kernel principal component analysis."
const ICA_DESCR = "Independent component analysis."
const PPCA_DESCR = "Probabilistic principal component analysis"
const FactorAnalysis_DESCR = "Factor Analysis"
const LDA_DESCR = """
      Multiclass linear discriminant analysis. The algorithm learns a
    projection matrix `P` that projects a feature matrix `Xtrain` onto a lower dimensional
    space of dimension `out_dim` such that the trace of the transformed between-class
    scatter matrix(`Pᵀ*Sb*P`) is maximized relative to the trace of the transformed
    within-class scatter matrix (`Pᵀ*Sw*P`).The projection matrix is scaled such that
    `Pᵀ*Sw*P=I` or `Pᵀ*Σw*P=I`(where `Σw` is the within-class covariance matrix) .
    Predicted class posterior probability for feature matrix `Xtest` are derived by
    applying a softmax transformationto a matrix `Pr`, such that  rowᵢ of `Pr` contains
    computed distances(based on a distance metric) in the transformed space of rowᵢ in
    `Xtest` to the centroid of each class.
    """
const BayesianLDA_DESCR = """
      Bayesian Multiclass linear discriminant analysis. The algorithm
    learns a projection matrix `P` that projects a feature matrix `Xtrain` onto a lower
    dimensional space of dimension `out_dim` such that the trace of the transformed
    between-class scatter matrix(`Pᵀ*Sb*P`) is maximized relative to the trace of the
    transformed within-class scatter matrix (`Pᵀ*Sw*P`). The projection matrix is scaled
    such that `Pᵀ*Sw*P = n` or `Pᵀ*Σw*P=I` (Where `n` is the number of training samples
    and `Σw` is the within-class covariance matrix).
    Predicted class posterior probability distibution are derived by applying Bayes rule
    with a multivariate Gaussian class-conditional distribution.
    """
const SubspaceLDA_DESCR = """
    Multiclass linear discriminant analysis. Suitable for high
    dimensional data (Avoids computing scatter matrices `Sw` ,`Sb`). The algorithm learns a
    projection matrix `P = W*L` that projects a feature matrix `Xtrain` onto a lower
    dimensional space of dimension `nc - 1` such that the trace of the transformed
    between-class scatter matrix(`Pᵀ*Sb*P`) is maximized relative to the trace of the
    transformed within-class scatter matrix (`Pᵀ*Sw*P`). The projection matrix is scaled
    such that `Pᵀ*Sw*P = mult*I` or `Pᵀ*Σw*P=mult/(n-nc)*I` (where `n` is the number of
    training samples, mult` is  one of `n` or `1` depending on whether `Sb` is normalized,
    `Σw` is the within-class covariance matrix, and `nc` is the number of unique classes
    in `y`) and also obeys `Wᵀ*Sb*p = λ*Wᵀ*Sw*p`, for every column `p` in `P`.
    Predicted class posterior probability for feature matrix `Xtest` are derived by
    applying a softmax transformation to a matrix `Pr`, such that  rowᵢ of `Pr` contains
    computed distances(based on a distance metric) in the transformed space of rowᵢ in
    `Xtest` to the centroid of each class.
    """
const BayesianSubspaceLDA_DESCR = """
       Bayesian Multiclass linear discriminant analysis. Suitable for high dimensional data
    (Avoids computing scatter matrices `Sw` ,`Sb`). The algorithm learns a projection
    matrix `P = W*L` (`Sw`), that projects a feature matrix `Xtrain` onto a lower
    dimensional space of dimension `nc-1` such that the trace of the transformed
    between-class scatter matrix(`Pᵀ*Sb*P`) is maximized relative to the trace of the
    transformed within-class scatter matrix (`Pᵀ*Sw*P`). The projection matrix is scaled
    such that `Pᵀ*Sw*P = mult*I` or `Pᵀ*Σw*P=mult/(n-nc)*I` (where `n` is the number of
    training samples, `mult` is  one of `n` or `1` depending on whether `Sb` is normalized,
    `Σw` is the within-class covariance matrix, and `nc` is the number of unique classes in
    `y`) and also obeys `Wᵀ*Sb*p = λ*Wᵀ*Sw*p`, for every column `p` in `P`.
    Posterior class probability distibution are derived by applying Bayes rule with a
    multivariate Gaussian class-conditional distribution
    """
const LinearRegressor_DESCR = """
    Linear Regression. Learns a linear combination of given
    variables to fit the response by minimizing the squared error between.
    """
const MultitargetLinearRegressor_DESCR = """
    Multitarget Linear Regression. Learns linear combinations of given
    variables to fit the responses by minimizing the squared error between.
    """
const RidgeRegressor_DESCR = """
    Ridge regressor with regularization parameter lambda. Learns a
    linear regression with a penalty on the l2 norm of the coefficients.
    """
const MultitargetRidgeRegressor_DESCR = """
    Multitarget Ridge regressor with regularization parameter lambda. Learns a
    Multitarget linear regression with a penalty on the l2 norm of the coefficients.
    """
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

`LinearRegressor` assumes the target is a continuous variable
whose conditional distribution is normal with constant variance, and whose
expected value is a linear combination of the features. Linear coefficients
are calculated using least squares.
Options exist to specify a bias term.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `schema(y)`

# Hyper-parameters

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new
  features `Xnew` having the same Scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `coefficients`: The linear coefficients determined by the model.
- `intercept`: The intercept determined by the model.

# Examples

```
using MLJ

LinearRegressor = @load LinearRegressor pkg=MultivariateStats
linear_regressor = LinearRegressor()


X, y = make_regression(100, 2) # synthetic data
mach = machine(linear_regressor, X, y) |> fit!

Xnew, _ = make_regression(3, 2)
yhat = predict(mach, Xnew) # new predictions
```

See also
TODO: ADD REFERENCES
"""
LinearRegressor

"""
$(MMI.doc_header(MultitargetLinearRegressor))

`MultitargetLinearRegressor` assumes the target is a continuous variable
whose conditional distribution is normal with constant variance, and whose
expected value is a linear combination of the features. Linear coefficients
are calculated using least squares. In this case, the output represents a
response vector.
Options exist to specify a bias term.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any table of responses whose element
  scitype is `Continuous`; check the scitype with `schema(y)`

# Hyper-parameters

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new
  features `Xnew` having the same Scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `coefficients`: The linear coefficients determined by the model.
- `intercept`: The intercept determined by the model.

# Examples

```
using MLJ
using MLJBase: augment_X
using DataFrames

LinearRegressor = @load MultitargetLinearRegressor pkg=MultivariateStats
linear_regressor = LinearRegressor()

X = augment_X(randn(100, 8), true)
θ = randn((9,2))
y = X * θ
X, y = map(x -> DataFrame(x, :auto), (X, y))

mach = machine(linear_regressor, X, y) |> fit!

Xnew, _ = make_regression(3, 9)
yhat = predict(mach, Xnew) # new predictions
```

See also
TODO: ADD REFERENCES
"""
MultitargetLinearRegressor

"""
$(MMI.doc_header(RidgeRegressor))

`RidgeRegressor` adds a quadratic penalty term to least squares regression,
for regularization. Ridge regression is particularly useful in the case of
multicollinearity.
Options exist to specify a bias term, and to adjust the strength of the penalty term.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `schema(y)`

# Hyper-parameters

- `lambda=1.0`: Is the non-negative parameter for the
  regularization strength. If lambda is 0, ridge regression is equivalent
  to linear least squares regression, and as lambda approaches infinity,
  all the linear coefficients approach 0.

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new
  features `Xnew` having the same Scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `coefficients`: The linear coefficients determined by the model.
- `intercept`: The intercept determined by the model.

# Examples

```
using MLJ

LinearRegressor = @load LinearRegressor pkg=MultivariateStats
RidgeRegressor = @load RidgeRegressor pkg=MultivariateStats

X, y = make_regression(100, 60) # synthetic data

linear_regressor = LinearRegressor()
mach = machine(linear_regressor, X, y) |> fit!
llsq_coef = fitted_params(mach).coefficients

ridge_regressor = RidgeRegressor(lambda=0)
ridge_mach = machine(ridge_regressor, X, y) |> fit!
coef = fitted_params(ridge_mach).coefficients
difference = llsq_coef - coef
@info "difference between λ=0 ridge and llsq" mean(difference) std(difference)


ridge_regressor = RidgeRegressor(lambda=1.5)
ridge_mach = machine(ridge_regressor, X, y) |> fit!

Xnew, _ = make_regression(3, 60)
yhat = predict(mach, Xnew) # new predictions
```

See also
TODO: ADD REFERENCES
"""
RidgeRegressor

"""
$(MMI.doc_header(MultitargetRidgeRegressor))

`MultitargetRidgeRegressor` adds a quadratic penalty term to least squares regression,
for regularization. Ridge regression is particularly useful in the case of
multicollinearity. In this case, the output represents a response vector.
Options exist to specify a bias term, and to adjust the strength of the penalty term.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X, y)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

- `y`: is the target, which can be any table of responses whose element
  scitype is `Continuous`; check the scitype with `schema(y)`

# Hyper-parameters

- `lambda=1.0`: Is the non-negative parameter for the
  regularization strength. If lambda is 0, ridge regression is equivalent
  to linear least squares regression, and as lambda approaches infinity,
  all the linear coefficients approach 0.

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new
  features `Xnew` having the same Scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `coefficients`: The linear coefficients determined by the model.
- `intercept`: The intercept determined by the model.

# Examples

```
using MLJ
using MLJBase: augment_X
using DataFrames

LinearRegressor = @load MultitargetLinearRegressor pkg=MultivariateStats
RidgeRegressor = @load MultitargetRidgeRegressor pkg=MultivariateStats

X = augment_X(randn(100, 80), true)
θ = randn((81,4))
y = X * θ
X, y = map(x -> DataFrame(x, :auto), (X, y))

# linear_regressor = LinearRegressor() # positive semi definite error for cholesky :(
# mach = machine(linear_regressor, X, y) |> fit!
# llsq_coef = fitted_params(mach).coefficients
#
# ridge_regressor = RidgeRegressor(lambda=0)
# ridge_mach = machine(ridge_regressor, X, y) |> fit!
# coef = fitted_params(ridge_mach).coefficients
# difference = llsq_coef - coef
# @info "difference between λ=0 ridge and llsq" mean(difference) std(difference)


ridge_regressor = RidgeRegressor(lambda=1.5)
ridge_mach = machine(ridge_regressor, X, y) |> fit!

Xnew, _ = make_regression(3, 60)
yhat = predict(mach, Xnew) # new predictions
```

See also
TODO: ADD REFERENCES
"""
MultitargetRidgeRegressor

"""
$(MMI.doc_header(PCA))

`PCA` Principal component analysis. Learns a linear transformation to
project the data  on a lower dimensional space while preserving most of the initial
variance.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

# Hyper-parameters

- `maxoutdim=0`: The maximum number of output dimensions. If not set, defaults to
  0, where all components are kept (e.g., the number of components/output dimensions
  is equal to the size of the smallest dimension of the training matrix)
- `method=:auto`: The method to use to solve the problem. Choices are
    - `:svd`: Support Vector Decomposition of the matrix.
    - `:cov`: Covariance matrix decomposition.
    - `:auto`: Use `:cov` if the matrices first dimension is smaller than its second dimension
      otherwise use `:svd`
- `pratio::Float64=0.99`: The ratio of variance preserved after the transformation
- `mean=nothing`: if set to nothing(default) centering will be computed and applied,
  if set to `0` no centering(assumed pre-centered), if a vector is passed,
  the centering is done with that vector.

# Operations

- `transform(mach, Xnew)`: Return lower dimensional projection of the target given new
  features `Xnew` having the same Scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projection`: Returns the projection matrix (of size `(d, p)`).
  Each column of the projection matrix corresponds to a principal component.
  The principal components are arranged in descending order of
  the corresponding variances.

# Report

The fields of `report(mach)` are:

- `indim`: Dimensions of the provided data.
- `outdim`: Dimensions of the transformed result.
- `tprincipalvar`: Total variance of the principal components.
- `tresidualvar`: Total residual variance.
- `tvar`: Total observation variance (principal + residual variance).
- `mean`: The mean vector (of length `d`).
- `principalvars`: The variance of the principal components.

# Examples

```
using MLJ

PCA = @load PCA pkg=MultivariateStats

X, y = @load_iris

model = PCA(maxoutdim=2)
mach = machine(model, X) |> fit!

projection = transform(mach, X)
```

See also
TODO: ADD REFERENCES
"""
PCA
"""
$(MMI.doc_header(KernelPCA))

`KernelPCA` Kernel principal component analysis. Using a kernel, the linear
operations of PCA are performed in a [reproducing Hilbert space](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space).

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

# Hyper-parameters

- `maxoutdim=0`: The maximum number of output dimensions. If not set, defaults to
  0, where all components are kept (e.g., the number of components/output dimensions
  is equal to the size of the smallest dimension of the training matrix).
- `kernel::Function=(x,y)->x'y`: The kernel function, takes in 2 vector arguments
   x and y, returns a scalar value. Defaults to the dot product of X and Y.
- `solver::Symbol=:auto`: solver to use for the eigenvalues, one of `:eig`(default),
  `:eigs`.
- `inverse::Bool=true`: perform calculations needed for inverse transform
- `beta::Real=1.0`: strength of the ridge regression that learns the inverse transform
  when inverse is true.
- `tol::Real=0.0`: Convergence tolerance for eigs solver.
- `maxiter::Int=300`: maximum number of iterations for eigs solver.

# Operations

- `transform(mach, Xnew)`: Return predictions of the target given new
  features `Xnew` having the same Scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projection`: Returns the projection matrix (of size `(d, p)`).
  Each column of the projection matrix corresponds to a principal component.
  The principal components are arranged in descending order of
  the corresponding variances.

# Report

The fields of `report(mach)` are:

- `indim`: Dimensions of the provided data.
- `outdim`: Dimensions of the transformed result.
- `principalvars`: The variance of the principal components.

# Examples

```
using MLJ
using LinearAlgebra

KPCA = @load KernelPCA pkg=MultivariateStats

X, y = @load_iris

function rbf_kernel(length_scale)
    return (x,y) -> norm(x-y)^2 / ((2 * length_scale)^2)
end

model = KPCA(maxoutdim=2, kernel = rbf_kernel(1))
mach = machine(model, X) |> fit!

projection = transform(mach, X)
```

See also
TODO: ADD REFERENCES
"""
KernelPCA
"""
$(MMI.doc_header(ICA))

`ICA` is a computational technique for separating a multivariate signal into
additive subcomponents, with the assumption that the subcomponents are
non-Gaussian and independent from each other.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

# Hyper-parameters

- `k::Int=0`: The number of independent components to recover, set automatically if `0`.
- `alg::Symbol=:fastica`: The algorithm to use (only `:fastica` is supported at the moment).
- `fun::Symbol=:tanh`: The approximate neg-entropy function, one of `:tanh`, `:gaus`.
- `do_whiten::Bool=true`: Whether or not to perform pre-whitening.
- `maxiter::Int=100`: The maximum number of iterations.
- `tol::Real=1e-6`: The convergence tolerance for change in matrix W.
- `mean::Union{Nothing, Real, Vector{Float64}}=nothing`: mean to use, if nothing (default)
   centering is computed and applied, if zero, no centering, a vector of means can
   be passed.
- `winit::Union{Nothing,Matrix{<:Real}}=nothing`: Initial guess for matrix `W` either
   an empty matrix (random initilization of `W`), a matrix of size `k × k` (if `do_whiten`
   is true), a matrix of size `m × k` otherwise. If unspecified i.e `nothing` an empty
   `Matrix{<:Real}` is used.

# Operations

- `transform(mach, Xnew)`: Return lower dimensional projection of the target given new
  features `Xnew` having the same Scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

 BUG: Does not have a projection class. It would also be cool to see the whitened
matrix in fitted_params, to show how the covariance is the identity

# Report

The fields of `report(mach)` are:

- `indim`: Dimensions of the provided data.
- `outdim`: Dimensions of the transformed result.
- `mean`: The mean vector.

# Examples

```
using MLJ
using LinearAlgebra

ICA = @load ICA pkg=MultivariateStats

X, y = @load_iris

model = ICA(k = 2, tol=0.1)
mach = machine(model, X) |> fit!

projection = transform(mach, X)
```

See also
TODO: ADD REFERENCES
"""
ICA
"""
$(MMI.doc_header(LDA))

`LDA`: Multiclass linear discriminant analysis. The algorithm learns a
projection matrix `P` that projects a feature matrix `Xtrain` onto a lower dimensional
space of dimension `out_dim` such that the trace of the transformed between-class
scatter matrix(`Pᵀ*Sb*P`) is maximized relative to the trace of the transformed
within-class scatter matrix (`Pᵀ*Sw*P`).The projection matrix is scaled such that
`Pᵀ*Sw*P=I` or `Pᵀ*Σw*P=I`(where `Σw` is the within-class covariance matrix) .
Predicted class posterior probability for feature matrix `Xtest` are derived by
applying a softmax transformationto a matrix `Pr`, such that  rowᵢ of `Pr` contains
computed distances(based on a distance metric) in the transformed space of rowᵢ in
`Xtest` to the centroid of each class.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)

Where

- `X`: is any table of input features (eg, a `DataFrame`) whose columns
  are of scitype `Continuous`; check the scitype with `schema(X)`

# Hyper-parameters

- `method::Symbol=:gevd`: The solver, one of `:gevd` or `:whiten` methods.
- `cov_w::CovarianceEstimator`=SimpleCovariance: An estimator for the within-class
    covariance (used in computing within-class scatter matrix, Sw), by default set
    to the standard `MultivariateStats.SimpleCovariance()` but
    could be set to any robust estimator from `CovarianceEstimation.jl`..
- `cov_b::CovarianceEstimator`=SimpleCovariance: The same as `cov_w` but for the between-class
    covariance (used in computing between-class scatter matrix, Sb).
- `out_dim::Int=0`: The output dimension, i.e dimension of the transformed space,
    automatically set if 0 is given (default).
- `regcoef::Float64=1e-6`: The regularization coefficient (default value 1e-6). A positive
    value `regcoef * eigmax(Sw)` where `Sw` is the within-class scatter matrix, is added
    to the diagonal of Sw to improve numerical stability. This can be useful if using
    the standard covariance estimator.
- `dist::SemiMetric=SqEuclidean`: The distance metric to use when performing classification
    (to compare the distance between a new point and centroids in the transformed space),
    an alternative choice can be the `CosineDist`.Defaults to `SqEuclidean`.

# Operations

- `transform(mach, Xnew)`: Return lower dimensional projection of the target given new
  features `Xnew` having the same Scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

 BUG: Does not have a projection class. It would also be cool to see the whitened
matrix in fitted_params, to show how the covariance is the identity

# Report

The fields of `report(mach)` are:

- `indim`: Dimensions of the provided data.
- `outdim`: Dimensions of the transformed result.
- `mean`: The mean vector.

# Examples

```
using MLJ
using LinearAlgebra

LA = @load LDA pkg=MultivariateStats

X, y = @load_iris

model = ICA(k = 2, tol=0.1)
mach = machine(model, X) |> fit!

projection = transform(mach, X)
```

See also
TODO: ADD REFERENCES
"""
LDA
BayesianLDA
SubspaceLDA
BayesianSubspaceLDA
FactorAnalysis
PPCA
end
