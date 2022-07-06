####
#### PCA
####

"""
    PCA(; kwargs...)

$PCA_DESCR

# Keyword Parameters

- `maxoutdim::Int=0`: maximum number of output dimensions, uses the smallest dimension of
    training feature matrix if 0 (default).
- `method::Symbol=:auto`: method to use to solve the problem, one of `:auto`,`:cov`
    or `:svd`
- `pratio::Float64=0.99`: ratio of variance preserved
- `mean::Union{Nothing, Real, Vector{Float64}}=nothing`: if set to nothing(default)
    centering will be computed and applied, if set to `0` no
    centering(assumed pre-centered), if a vector is passed, the centering is done with
    that vector.
"""
@mlj_model mutable struct PCA <: MMI.Unsupervised
    maxoutdim::Int = 0::(_ ≥ 0)
    method::Symbol = :auto::(_ in (:auto, :cov, :svd))
    pratio::Float64 = 0.99::(0.0 < _ ≤ 1.0)
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
        pratio=model.pratio,
        maxoutdim=maxoutdim,
        mean=model.mean
    )
    cache = nothing
    report = (
        indim=MS.size(fitresult,1)
        outdim=MS.size(fitresult,2),
        tprincipalvar=MS.tprincipalvar(fitresult),
        tresidualvar=MS.tresidualvar(fitresult),
        tvar=MS.var(fitresult),
        mean=copy(MS.mean(fitresult)),
        principalvars=copy(MS.principalvars(fitresult))
    )
    return fitresult, cache, report
end

metadata_model(PCA,
    input=Table(Continuous),
    output=Table(Continuous),
    weights=false,
    descr=PCA_DESCR,
    path="$(PKG).PCA"
)

####
#### KernelPCA
####

"""
    KernelPCA(; kwargs...)

$KPCA_DESCR

# Keyword Parameters

- `maxoutdim::Int = 0`: maximum number of output dimensions, uses the smallest
    dimension of training feature matrix if 0 (default).
- `kernel::Function=(x,y)->x'y`: kernel function of 2 vector arguments x and y, returns a
    scalar value
- `solver::Symbol=:auto`: solver to use for the eigenvalues, one of `:eig`(default),
    `:eigs`
- `inverse::Bool=true`: perform calculations needed for inverse transform
- `beta::Real=1.0`: strength of the ridge regression that learns the inverse transform
    when inverse is true
- `tol::Real=0.0`: Convergence tolerance for eigs solver
- `maxiter::Int=300`: maximum number of iterations for eigs solver
"""
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
        indim=MS.size(fitresult,1),
        outdim=MS.size(fitresult,2),
        principalvars=copy(MS.eigvals(fitresult))
    )
    return fitresult, cache, report
end

metadata_model(
    KernelPCA,
    input=Table(Continuous),
    output=Table(Continuous),
    weights=false,
    descr=KPCA_DESCR,
    path="$(PKG).KernelPCA"
)


####
#### ICA
####

"""
    ICA(; kwargs...)

$ICA_DESCR

# Keyword Parameters

- `k::Int=0`: number of independent components to recover, set automatically if `0`
- `alg::Symbol=:fastica`: algorithm to use (only `:fastica` is supported at the moment)
- `fun::Symbol=:tanh`: approximate neg-entropy function, one of `:tanh`, `:gaus`
- `do_whiten::Bool=true`: whether to perform pre-whitening
- `maxiter::Int=100`: maximum number of iterations
- `tol::Real=1e-6`: convergence tolerance for change in matrix W
- `mean::Union{Nothing, Real, Vector{Float64}}=nothing`: mean to use, if nothing (default)
    centering is computed andapplied, if zero, no centering, a vector of means can
    be passed
- `winit::Union{Nothing,Matrix{<:Real}}=nothing`: initial guess for matrix `W` either
    an empty matrix (random initilization of `W`), a matrix of size `k × k` (if `do_whiten`
    is true), a matrix of size `m × k` otherwise. If unspecified i.e `nothing` an empty
    `Matrix{<:Real}` is used.
"""
@mlj_model mutable struct ICA <: MMI.Unsupervised
    k::Int = 0::(_ ≥ 0)
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
    k = ifelse(model.k ≤ m, model.k, m)
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
        indim=MS.size(fitresult,1),
        outdim=MS.size(fitresult,2),
        mean=copy(MS.mean(fitresult))
    )
    return fitresult, cache, report
end

metadata_model(
    ICA,
    input=Table(Continuous),
    output=Table(Continuous),
    weights=false,
    descr=ICA_DESCR,
    path="$(PKG).ICA"
)

####
#### PPCA
####

"""
    PPCA(; kwargs...)

$PPCA_DESCR

# Keyword Parameters

- `maxoutdim::Int=0`: maximum number of output dimensions, uses max(no_of_features - 1, 1)
    if 0 (default).
- `method::Symbol=:ml`: method to use to solve the problem, one of `:ml`, `:em`, `:bayes`.
- `maxiter::Int=1000`: maximum number of iterations.
- `tol::Real=1e-6`: convergence tolerance.
- `mean::Union{Nothing, Real, Vector{Float64}}=nothing`: if set to nothing(default)
    centering will be computed and applied, if set to `0` no
    centering(assumed pre-centered), if a vector is passed, the centering is done with
    that vector.
"""
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
        indim=MS.size(fitresult,1),
        outdim=MS.size(fitresult,2),
        tvar=MS.var(fitresult),
        mean=copy(MS.mean(fitresult)),
        loadings=MS.loadings(fitresult)
    )
    return fitresult, cache, report
end

metadata_model(PPCA,
    input=Table(Continuous),
    output=Table(Continuous),
    weights=false,
    descr=PPCA_DESCR,
    path="$(PKG).PPCA"
)

####
#### FactorAnalysis
####

"""
    FactorAnalysis(; kwargs...)

$PPCA_DESCR

# Keyword Parameters

- `method::Symbol=:cm`: Method to use to solve the problem, one of `:ml`, `:em`, `:bayes`.
- `maxoutdim::Int=0`: Maximum number of output dimensions, uses max(no_of_features - 1, 1)
    if 0 (default).
- `maxiter::Int=1000`: Maximum number of iterations.
- `tol::Real=1e-6`: Convergence tolerance.
- `eta::Real=tol`: Variance lower bound
- `mean::Union{Nothing, Real, Vector{Float64}}=nothing`: If set to nothing(default)
    centering will be computed and applied, if set to `0` no
    centering(assumed pre-centered), if a vector is passed, the centering is done with
    that vector.
"""
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
        indim=MS.size(fitresult,1),
        outdim=MS.size(fitresult,2),
        variance=MS.var(fitresult),
        covariance_matrix=MS.cov(fitresult),
        mean=MS.mean(fitresult),
        loadings=MS.loadings(fitresult)
    )
    return fitresult, cache, report
end

metadata_model(FactorAnalysis,
    input=Table(Continuous),
    output=Table(Continuous),
    weights=false,
    descr=FactorAnalysis_DESCR,
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
    @eval function MMI.fitted_params(::$M, fr)
        return (projection=copy(MS.projection(fr)),)
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
