####
#### PCA
####

@mlj_model mutable struct PCA <: MMI.Unsupervised
    maxoutdim::Int = 0::(_ ≥ 0)
    method::Symbol = :auto::(_ in (:auto, :cov, :svd))
    variance_ratio::Float64 = 0.99::(0.0 < _ ≤ 1.0)
    mean::Union{Nothing, Real, Vector{Float64}} = nothing::(_check_typeof_mean(_))
end

function _check_typeof_mean(x)
    return x isa Vector{<:Real} || x === nothing || (x isa Real && iszero(x))
end

function MMI.fit(model::PCA, verbosity::Int, X)
    Xarray = MMI.matrix(X)
    mindim = minimum(size(Xarray))
    maxoutdim = model.maxoutdim == 0 ? mindim : model.maxoutdim
    fitresult = MS.fit(
        MS.PCA, Xarray';
        method=model.method,
        pratio=model.variance_ratio,
        maxoutdim=maxoutdim,
        mean=model.mean
    )
    cache = nothing
    report = (
        indim=size(fitresult)[1],
        outdim=size(fitresult)[2],
        tprincipalvar=MS.tprincipalvar(fitresult),
        tresidualvar=MS.tresidualvar(fitresult),
        tvar=MS.var(fitresult),
        mean=copy(MS.mean(fitresult)),
        principalvars=copy(MS.principalvars(fitresult)),
        # no need to copy here as a new copy is created 
        # for each function call
        loadings = MS.loadings(fitresult) 
    )
    return fitresult, cache, report
end

metadata_model(PCA,
    input=Table(Continuous),
    output=Table(Continuous),
    weights=false,
    path="$(PKG).PCA"
)

####
#### KernelPCA
####

@mlj_model mutable struct KernelPCA <: MMI.Unsupervised
    maxoutdim::Int = 0::(_ ≥ 0)
    kernel::Union{Nothing, Function} = default_kernel
    solver::Symbol = :eig::(_ in (:eig, :eigs))
    inverse::Bool = true
    beta::Real = 1.0::(_ ≥ 0.0)
    tol::Real = 1e-6::(_ ≥ 0.0)
    maxiter::Int = 300::(_ ≥ 1)
end

function MMI.fit(model::KernelPCA, verbosity::Int, X)
    Xarray = MMI.matrix(X)
    mindim = minimum(size(Xarray))
    # default max out dim if not given
    maxoutdim = model.maxoutdim == 0 ? mindim : model.maxoutdim
    fitresult = MS.fit(
        MS.KernelPCA,
        permutedims(Xarray);
        kernel=model.kernel,
        maxoutdim=maxoutdim,
        solver=model.solver,
        inverse=model.inverse,
        β=model.beta,tol=model.tol,
        maxiter=model.maxiter
    )
    cache  = nothing
    report = (
        indim=size(fitresult)[1],
        outdim=size(fitresult)[2],
        principalvars=copy(MS.eigvals(fitresult))
    )
    return fitresult, cache, report
end

metadata_model(
    KernelPCA,
    human_name="kernel prinicipal component analysis model",
    input=Table(Continuous),
    output=Table(Continuous),
    weights=false,
    path="$(PKG).KernelPCA"
)


####
#### ICA
####

@mlj_model mutable struct ICA <: MMI.Unsupervised
    outdim::Int = 0::(_ ≥ 0)
    alg::Symbol = :fastica::(_ in (:fastica,))
    fun::Symbol = :tanh::(_ in (:tanh, :gaus))
    do_whiten::Bool = true
    maxiter::Int=100::(_ ≥ 1)
    tol::Real = 1e-6::(_ ≥ 0.0)
    winit::Union{Nothing, Matrix{<:Real}}= nothing
    mean::Union{Nothing, Real, Vector{Float64}} = nothing::(_check_typeof_mean(_))
end

function MMI.fit(model::ICA, verbosity::Int, X)
    icagfun(fname::Symbol, ::Type{T} = Float64) where T<:Real=
    fname == :tanh ? MS.Tanh{T}(1.0) :
    fname == :gaus ? MS.Gaus() :
    error("Unknown gfun $(fname)")

    Xarray = MMI.matrix(X)
    n, p = size(Xarray)
    m = min(n, p)
    k = ifelse(model.outdim ≤ m, model.outdim, m)
    fitresult = MS.fit(
        MS.ICA, Xarray', k;
        alg=model.alg,
        fun=icagfun(model.fun, eltype(Xarray)),
        do_whiten=model.do_whiten,
        maxiter=model.maxiter,
        tol=model.tol,
        mean=model.mean,
        winit=model.winit === nothing ? zeros(eltype(Xarray), 0, 0) : model.winit
    )
    cache = nothing
    report = (
        indim=size(fitresult)[1],
        outdim=size(fitresult)[2],
        mean=copy(MS.mean(fitresult))
    )
    return fitresult, cache, report
end

metadata_model(
    ICA,
    human_name="independent component analysis model",
    input=Table(Continuous),
    output=Table(Continuous),
    weights=false,
    path="$(PKG).ICA"
)

####
#### PPCA
####

@mlj_model mutable struct PPCA <: MMI.Unsupervised
    maxoutdim::Int = 0::(_ ≥ 0)
    method::Symbol = :ml::(_ in (:ml, :em, :bayes))
    maxiter::Int = 1000
    tol::Real = 1e-6::(_ ≥ 0.0)
    mean::Union{Nothing, Real, Vector{Float64}} = nothing::(_check_typeof_mean(_))
end

function MMI.fit(model::PPCA, verbosity::Int, X)
    Xarray = MMI.matrix(X)
    def_dim = max(1, size(Xarray, 2) - 1)
    maxoutdim = model.maxoutdim == 0 ? def_dim : model.maxoutdim
    fitresult = MS.fit(
        MS.PPCA, Xarray';
        method=model.method,
        tol=model.tol,
        maxiter=model.maxiter,
        maxoutdim=maxoutdim,
        mean=model.mean
    )
    cache = nothing
    report = (
        indim=size(fitresult)[1],
        outdim=size(fitresult)[2],
        tvar=MS.var(fitresult),
        mean=copy(MS.mean(fitresult)),
        loadings = copy(MS.loadings(fitresult))
    )
    return fitresult, cache, report
end

metadata_model(
    PPCA,
    human_name="probabilistic PCA model",
    input=Table(Continuous),
    output=Table(Continuous),
    weights=false,
    path="$(PKG).PPCA"
)

####
#### FactorAnalysis
####

@mlj_model mutable struct FactorAnalysis <: MMI.Unsupervised
    method::Symbol=:cm::(_ in (:em, :cm))
    maxoutdim::Int=0::(_ ≥ 0)
    maxiter::Int=1000::(_ ≥ 1)
    tol::Real=1e-6::(_ ≥ 0.0)
    eta::Real=tol::(_ ≥ 0.0)
    mean::Union{Nothing, Real, Vector{Float64}} = nothing::(_check_typeof_mean(_))
end

function MMI.fit(model::FactorAnalysis, verbosity::Int, X)
    Xarray = MMI.matrix(X)
    def_dim = max(1, size(Xarray, 2) - 1)
    maxoutdim = model.maxoutdim == 0 ? def_dim : model.maxoutdim
    fitresult = MS.fit(
        MS.FactorAnalysis, Xarray';
        method=model.method,
        maxiter=model.maxiter,
        tol=model.tol,
        η=model.eta,
        maxoutdim=maxoutdim,
        mean=model.mean
    )
    cache = nothing
    report = (
        indim=size(fitresult)[1],
        outdim=size(fitresult)[2],
        variance=MS.var(fitresult),
        covariance_matrix=MS.cov(fitresult),
        mean=MS.mean(fitresult),
        loadings=MS.loadings(fitresult)
    )
    return fitresult, cache, report
end

metadata_model(
    FactorAnalysis,
    human_name="factor analysis model",
    input=Table(Continuous),
    output=Table(Continuous),
    weights=false,
    path="$(PKG).FactorAnalysis"
)

####
#### Common interface
####

model_types = [
    (PCA, PCAFitResultType),
    (KernelPCA, KernelPCAFitResultType),
    (ICA, ICAFitResultType),
    (PPCA, PPCAFitResultType),
    (FactorAnalysis, FactorAnalysisResultType)
]

for (M, MFitResultType) in model_types

    if M !== ICA # special cased below
        quote
            function MMI.fitted_params(::$M, fr)
                return (projection=copy(MS.projection(fr)),)
            end
        end |> eval
    end

    @eval function MMI.transform(::$M, fr::$MFitResultType, X)
        # X is n x d, need to take adjoint twice
        Xarray = MMI.matrix(X)
        Xnew = MS.predict(fr, Xarray')'
        return MMI.table(Xnew, prototype=X)
    end

    if hasmethod(MS.reconstruct, Tuple{MFitResultType{Float64}, Matrix{Float64}})
        @eval function MMI.inverse_transform(::$M, fr::$MFitResultType, Y)
            # X is n x p, need to take adjoint twice
            Yarray = MMI.matrix(Y)
            Ynew = MS.reconstruct(fr, Yarray')'
            return MMI.table(Ynew, prototype=Y)
        end
    end
end

MMI.fitted_params(::ICA, fr) = (projection=copy(fr.W), mean = copy(MS.mean(fr)))


# # DOCUMENT STRINGS

"""

$(MMI.doc_header(PCA))

Principal component analysis learns a linear projection onto a lower dimensional space 
while preserving most of the initial variance seen in the training data.

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

- `indim`: Dimension (number of columns) of the training data and new data to be 
  transformed.

- `outdim = min(n, indim, maxoutdim)` is the output dimension; here `n` is the number of
  observations.

- `tprincipalvar`: Total variance of the principal components.

- `tresidualvar`: Total residual variance.

- `tvar`: Total observation variance (principal + residual variance).

- `mean`: The mean of the untransformed training data, of length `indim`.

- `principalvars`: The variance of the principal components. An AbstractVector of 
  length `outdim`

- `loadings`: The models loadings, weights for each variable used when calculating 
  principal components. A matrix of size (`indim`, `outdim`) where `indim` and 
  `outdim` are as defined above.

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
  orthogonal to the kernel of that map. In particular, if 
  `Xsmall = transform(mach, Xnew)`, then `inverse_transform(Xsmall)` is only an 
  approximation to `Xnew`.

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

- `outdim::Int=0`: The number of independent components to recover, set automatically 
  if `0`.

- `alg::Symbol=:fastica`: The algorithm to use (only `:fastica` is supported at the 
  moment).

- `fun::Symbol=:tanh`: The approximate neg-entropy function, one of `:tanh`, `:gaus`.

- `do_whiten::Bool=true`: Whether or not to perform pre-whitening.

- `maxiter::Int=100`: The maximum number of iterations.

- `tol::Real=1e-6`: The convergence tolerance for change in the unmixing matrix W.

- `mean::Union{Nothing, Real, Vector{Float64}}=nothing`: mean to use, if nothing (default)
  centering is computed and applied, if zero, no centering; otherwise a vector of means 
  can be passed.

- `winit::Union{Nothing,Matrix{<:Real}}=nothing`: Initial guess for the unmixing matrix 
  `W`: either an empty matrix (for random initialization of `W`), a matrix of size 
  `m × k` (if `do_whiten` is true), or a matrix of size `m × k`. Here `m` is the number 
  of components (columns) of the input.

# Operations

- `transform(mach, Xnew)`: Return the component-separated version of input `Xnew`, which 
  should have the same scitype as `X` above.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projection`: The estimated component matrix.

- `mean`: The estimated mean vector.

# Report

The fields of `report(mach)` are:

- `indim`: Dimension (number of columns) of the training data and new data to be 
  transformed.

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

$(MMI.doc_header(FactorAnalysis))

Factor analysis is a linear-Gaussian latent variable model that is closely related to
probabilistic PCA. In contrast to the probabilistic PCA model, the covariance of 
conditional distribution of the observed variable given the latent variable is diagonal 
rather than isotropic.

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

- `indim`: Dimension (number of columns) of the training data and new data to be 
  transformed.

- `outdim`: Dimension of transformed data (number of factors).

- `variance`: The variance of the factors.

- `covariance_matrix`: The estimated covariance matrix.

- `mean`: The mean of the untransformed training data, of length `indim`.

- `loadings`: The factor loadings. A matrix of size (`indim`, `outdim`) where 
  `indim` and `outdim` are as defined above.

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
  `Xsmall = transform(mach, Xnew)`, then `inverse_transform(Xsmall)` is only an 
  approximation to `Xnew`.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `projection`: Returns the projection matrix, which has size `(indim, outdim)`, where
  `indim` and `outdim` are the number of features of the input and ouput respectively.
  Each column of the projection matrix corresponds to a principal component.

# Report

The fields of `report(mach)` are:

- `indim`: Dimension (number of columns) of the training data and new data to be 
  transformed.

- `outdim`: Dimension of transformed data.

- `tvat`: The variance of the components.

- `loadings`: The model's loadings matrix. A matrix of size (`indim`, `outdim`) where 
  `indim` and `outdim` as as defined above.

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
