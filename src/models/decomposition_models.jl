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
        principalvars=copy(MS.principalvars(fitresult))
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
        loadings=MS.loadings(fitresult)
    )
    return fitresult, cache, report
end

metadata_model(PPCA,
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

metadata_model(FactorAnalysis,
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
