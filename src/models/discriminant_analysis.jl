####
#### MulticlassLDA
####
"""
    LDA(; kwargs...)

$LDA_DESCR

# Keyword Parameters

- `method::Symbol=:gevd`:  choice of solver, one of `:gevd` or `:whiten` methods
- `cov_w::CovarianceEstimator`=SimpleCovariance: an estimator for the within-class 
    covariance (used in computing within-class scatter matrix, Sw), by default set
    to the standard `MultivariateStats.SimpleCovariance()` but 
    could be set to any robust estimator from `CovarianceEstimation.jl`.
- `cov_b::CovarianceEstimator`=SimpleCovariance: same as `cov_w` but for the between-class 
    covariance (used in computing between-class scatter matrix, Sb)
- `out_dim::Int=0`: the output dimension, i.e dimension of the transformed space, 
    automatically set if 0 is given (default).
- `regcoef::Float64=1e-6`: regularization coefficient (default value 1e-6). A positive 
    value `regcoef * eigmax(Sw)` where `Sw` is the within-class scatter matrix, is added 
    to the diagonal of Sw to improve numerical stability. This can be useful if using 
    the standard covariance estimator.
- `dist::SemiMetric=SqEuclidean`: the distance metric to use when performing classification
    (to compare the distance between a new point and centroids in the transformed space), 
    an alternative choice can be the `CosineDist`.Defaults to `SqEuclidean`

See also the 
[package documentation](https://multivariatestatsjl.readthedocs.io/en/latest/lda.html).
For more information about the algorithm, see the paper by Li, Zhu and Ogihara, 
[Using Discriminant Analysis for Multi-class Classification: 
An Experimental Investigation](http://citeseerx.ist.psu.edu/viewdoc/
download?doi=10.1.1.89.7068&rep=rep1&type=pdf).
"""
@mlj_model mutable struct LDA <: MMI.Probabilistic
    method::Symbol = :gevd::(_ in (:gevd, :whiten))
    cov_w::CovarianceEstimator = MS.SimpleCovariance()
    cov_b::CovarianceEstimator = MS.SimpleCovariance()
    out_dim::Int = 0::(_ ≥ 0)
    regcoef::Float64 = 1e-6::(_ ≥ 0)
    dist::SemiMetric = SqEuclidean()
end

function MMI.fit(model::LDA, ::Int, X, y)
    Xm_t, yplain, classes_seen, p, n, nc, nclasses, integers_seen, out_dim = 
        _check_lda_data(model, X, y)
    core_res = MS.fit(
        MS.MulticlassLDA, nc, Xm_t, Int.(yplain);
        method=model.method,
        outdim=out_dim,
        regcoef=model.regcoef,
        covestimator_within=model.cov_w,
        covestimator_between=model.cov_b
    )
    cache = nothing
    report = (
        classes=classes_seen,
        out_dim=MS.size(core_res)[2],
        class_means=MS.classmeans(core_res),
        mean=MS.mean(core_res),
        class_weights=MS.classweights(core_res),
        Sw=MS.withclass_scatter(core_res),
        Sb=MS.betweenclass_scatter(core_res),
        nc=nc
    )
    fitresult = (core_res, classes_seen)
    return fitresult, cache, report
end

function _check_lda_data(model, X, y)
    class_list = MMI.classes(y[1]) # Class list containing entries in pool of y.
    nclasses = length(class_list)
    # Class list containing entries in seen in y.
    classes_seen = filter(in(y), class_list) 
    nc = length(classes_seen) # Number of classes in pool of y.
    integers_seen = MMI.int(classes_seen)
    # NOTE: copy/transpose.
    Xm_t = _matrix_transpose(model, X) # Now p x n matrix
    yplain = MMI.int(y) # Vector of n ints in {1,..., nclasses}.
    p, n = size(Xm_t)
    # Recode yplain to be in {1,..., nc}
    nc == nclasses || _replace!(yplain, integers_seen, 1:nc) 
    # Check to make sure we have more than one class in training sample.
    # This is to prevent Sb from being a zero matrix.
    if nc <= 1
        throw(
            ArgumentError(
                "The number of unique classes in "*
                "traning sample has to be greater than one"
            )
        )
    end
    # Check to make sure we have more samples than classes.
    # This is to prevent Sw from being the zero matrix.
    if n <= nc
        throw(
            ArgumentError(
                "The number of training samples `n` has"*
                " to be greater than number of unique classes `nc`"
            )
        )
    end
    # Check output dimension default is min(p, nc-1)
    def_outdim = min(p, nc - 1)
    # If unset (0) use the default; otherwise try to use the provided one
    out_dim = ifelse(model.out_dim == 0, def_outdim, model.out_dim)
    # Check if the given one is sensible
    if out_dim > p
        throw(
            ArgumentError(
                "`out_dim` must not be larger than `p`"*
                "where `p` is the number of features in `X`"
            )
        )
    end 
    return Xm_t, yplain, classes_seen, p, n, nc, nclasses, integers_seen, out_dim
end

function MMI.fitted_params(::LDA, (core_res, classes_seen))
    return (
        projected_class_means=MS.classmeans(core_res),
        projection_matrix=MS.projection(core_res)
    )
end

function MMI.predict(m::LDA, (core_res, classes_seen), Xnew)
    # projection of Xnew, XWt is nt x o  where o = number of out dims
    # nt = number ot test samples
    XWt = MMI.matrix(Xnew) * core_res.proj
    # centroids in the transformed space, nc x o
    centroids = transpose(core_res.pmeans)

    # compute the distances in the transformed space between pairs of rows
    # the probability matrix Pr is `n x nc` and normalised accross rows
    Pr = pairwise(m.dist, XWt, centroids, dims=1)
    Pr .*= -1
    # apply a softmax transformation
    softmax!(Pr)
    return MMI.UnivariateFinite(classes_seen, Pr)
end

metadata_model(LDA,
    input=Table(Continuous),
    target=AbstractVector{<:Finite},
    weights=false,
    output=Table(Continuous),
    descr=LDA_DESCR,
    path="$(PKG).LDA"
)
    

####
#### BayesianLDA
####

"""
    BayesianLDA(; kwargs...)

$BayesianLDA_DESCR

# Keyword Parameters

- `method::Symbol=:gevd`: choice of solver, one of `:gevd` or `:whiten` methods
- `cov_w::CovarianceEstimator=SimpleCovariance()`: an estimator for the within-class 
    covariance (used in computing within-class scatter matrix, Sw), by default set to the 
    standard `MultivariateStats.CovarianceEstimator` but could be set to any robust 
    estimator from `CovarianceEstimation.jl`.
- `cov_b::CovarianceEstimator=SimpleCovariance()`: same as `cov_w` but for the  
    between-class covariance(used in computing between-class scatter matrix, Sb).
- `out_dim::Int=0`: the output dimension, i.e dimension of the transformed space, 
    automatically set if 0 is given (default).
- `regcoef::Float64=1e-6`: regularization coefficient (default value 1e-6). A positive 
value `regcoef * eigmax(Sw)` where `Sw` is the within-class covariance estimator, is added 
    to the diagonal of Sw to improve numerical stability. This can be useful if using the 
    standard covariance estimator.
- `priors::Union{Nothing, Vector{Float64}}=nothing`: For use in prediction with Baye's rule. If `priors = nothing` then 
    `priors` are estimated from the class proportions in the training data. Otherwise it 
    requires a `Vector` containing class probabilities with probabilities specified using 
    the order given by `levels(y)` where y is the target vector.

See also the [package documentation](
https://multivariatestatsjl.readthedocs.io/en/latest/lda.html).
For more information about the algorithm, see the paper by Li, Zhu and Ogihara, 
[Using Discriminant Analysis for Multi-class Classification: An Experimental Investigation](
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.7068&rep=rep1&type=pdf).
"""
@mlj_model mutable struct BayesianLDA <: MMI.Probabilistic
    method::Symbol = :gevd::(_ in (:gevd, :whiten))
    cov_w::CovarianceEstimator=MS.SimpleCovariance()
    cov_b::CovarianceEstimator=MS.SimpleCovariance()
    out_dim::Int=0::(_ ≥ 0)
    regcoef::Float64=1e-6::(_ ≥ 0)
    priors::Union{Nothing, Vector{Float64}}=nothing
end

function MMI.fit(model::BayesianLDA, ::Int, X, y)
    Xm_t, yplain, classes_seen, p, n, nc, nclasses, integers_seen, out_dim = 
        _check_lda_data(model, X, y) 
    ## If piors are specified check if they makes sense.
    ## This was put here to through errors much earlier
    if isa(model.priors, Vector)
        priors = _check_lda_priors(model.priors, nc, nclasses, integers_seen)         
    end
 
    core_res = MS.fit(
        MS.MulticlassLDA, nc, Xm_t, Int.(yplain);
        method=model.method,
        outdim=out_dim,
        regcoef=model.regcoef,
        covestimator_within=model.cov_w,
        covestimator_between=model.cov_b
    )

    ## Estimates prior probabilities if specified by user.
    ## Put it here to avoid recomputing as the fitting process does this already
    if isa(model.priors, Nothing)
        weights = MS.classweights(core_res)
        total = core_res.stats.tweight
        priors = weights ./ total
    end
    cache     = nothing
    report    = (
        classes=classes_seen,
        out_dim=MS.size(core_res)[2],
        class_means=MS.classmeans(core_res),
        mean=MS.mean(core_res),
        class_weights=MS.classweights(core_res),
        Sw=MS.withclass_scatter(core_res),
        Sb=MS.betweenclass_scatter(core_res),
        nc=nc
    )

    fitresult = (core_res, classes_seen, priors, n)
    return fitresult, cache, report
end

function _matrix_transpose(model::Union{LDA, BayesianLDA}, X)
    return MMI.matrix(X; transpose=true)
end

@inline function _check_lda_priors(priors, nc, nclasses, integers_seen)
    if length(priors) != nclasses
        throw(ArgumentError("Invalid size of `priors`."))
    end
     
    # `priors` is esssentially always an instance of type `Vector{Float64}`.
    # The next two conditions implicitly checks that
    # ` 0 .<= priors .<= 1` and `sum(priors) ≈ 1` are true.
    if !isapprox(sum(priors), 1)
        throw(ArgumentError("probabilities specified in `priors` must sum to 1"))
    end
    if all(>=(0), priors) 
        throw(ArgumentError("probabilities specified in `priors` must non-negative"))
    end
    # Select priors for unique classes in `y` (For resampling purporses).
    priors_ = nc == nclasses ? model.priors : @view model.priors[integers_seen]
    return priors_
end

_get_priors(priors::SubArray) = copy(priors)
_get_priors(priors) = priors

function MMI.fitted_params(::BayesianLDA, (core_res, classes_seen, priors, n))
   return (
       projected_class_means=MS.classmeans(core_res),
       projection_matrix=MS.projection(core_res),
       priors=_get_priors(priors)
    )
end

function MMI.predict(m::BayesianLDA, (core_res, classes_seen, priors, n), Xnew)
     # projection of Xnew, XWt is nt x o  where o = number of out dims
    # nt = number ot test samples
    XWt = MMI.matrix(Xnew) * core_res.proj
    # centroids in the transformed space, nc x o
    centroids = transpose(core_res.pmeans)
  
    # The discriminant matrix `Pr` is of dimension `nt x nc`
    # Pr[i,k] = -0.5*(xᵢ −  µₖ)ᵀ(Σw⁻¹)(xᵢ −  µₖ) + log(priorsₖ) where (Σw = Sw/n)
    # In the transformed space this becomes
    # Pr[i,k] = -0.5*(Pᵀxᵢ −  Pᵀµₖ)ᵀ(PᵀΣw⁻¹P)(Pᵀxᵢ −  Pᵀµₖ) + log(priorsₖ)
    # But PᵀSw⁻¹P = I and PᵀΣw⁻¹P = n*I due to the nature of the projection_matrix, P
    # Giving Pr[i,k] = -0.5*n*(Pᵀxᵢ −  Pᵀµₖ)ᵀ(Pᵀxᵢ −  Pᵀµₖ) + log(priorsₖ)
    # with (Pᵀxᵢ −  Pᵀµₖ)ᵀ(Pᵀxᵢ −  Pᵀµₖ) being the SquaredEquclidean distance between
    # pairs of rows in the transformed space
    Pr = pairwise(SqEuclidean(), XWt, centroids, dims=1)
    Pr .*= (-n/2)
    Pr .+= log.(transpose(priors))

    # apply a softmax transformation to convert Pr to a probability matrix
    softmax!(Pr)
    return MMI.UnivariateFinite(classes_seen, Pr)
end

function MMI.transform(m::T, (core_res, ), X) where T<:Union{LDA, BayesianLDA}
    # projection of X, XWt is nt x o  where o = out dims
    proj = core_res.proj #proj is the projection_matrix
    XWt = MMI.matrix(X) * proj
    return MMI.table(XWt, prototype = X)
end

metadata_model(
    BayesianLDA,
    input=Table(Continuous),
    target= AbstractVector{<:Finite},
    weights=false,
    output=Table(Continuous),
    descr=BayesianLDA_DESCR,
    path="$(PKG).BayesianLDA"
)
    
####
#### SubspaceLDA
####

"""
    SubspaceLDA(; kwargs...)

$SubspaceLDA_DESCR

# Keyword Parameters

- `normalize=true`: Option to normalize the between class variance for the number of 
    observations in each class, one of `true` or `false`.
- `out_dim`: the dimension of the transformed space to be used by `predict` and 
    `transform` methods, automatically set if 0 is given (default).
- `dist=SqEuclidean`: the distance metric to use when performing classification 
    (to compare the distance between a new point and centroids in the transformed space), 
    an alternative choice can be the `CosineDist`.

See also the [package documentation](
https://multivariatestatsjl.readthedocs.io/en/latest/lda.html).
For more information about the algorithm, see the paper by Howland & Park (2006),
"Generalizing discriminant analysis using the generalized singular value decomposition",
IEEE Trans. Patt. Anal. & Mach. Int., 26: 995-1006.
"""
@mlj_model mutable struct SubspaceLDA <: MMI.Probabilistic
    normalize::Bool = true
    out_dim::Int = 0::(_ ≥ 0)
    dist::SemiMetric = SqEuclidean()
end

function MMI.fit(model::SubspaceLDA, ::Int, X, y)
    Xm_t, yplain, classes_seen, p, n, nc, nclasses, integers_seen, out_dim =
        _check_lda_data(model, X, y) 

    core_res = MS.fit(
        MS.SubspaceLDA, Xm_t, Int.(yplain), nc;
        normalize = model.normalize
    )
    # λ is a (nc -1) x 1 vector containing the eigen values sorted in descending order.
    λ = core_res.λ 
    explained_variance_ratio = λ ./ sum(λ) #proportions of variance

    cache = nothing
    report = (
        explained_variance_ratio=explained_variance_ratio,
        classes=classes_seen,
        class_means=MS.classmeans(core_res),
        mean=MS.mean(core_res),
        class_weights=MS.classweights(core_res),
        nc=nc
    )
    fitresult = (core_res, out_dim, classes_seen)
    return fitresult, cache, report
end

function MMI.fitted_params(::SubspaceLDA, (core_res, _))
    return (class_means=MS.classmeans(core_res), projection_matrix=MS.projection(core_res))
end

function MMI.predict(m::SubspaceLDA, (core_res, out_dim, classes_seen), Xnew)
     # projection of Xnew, XWt is nt x o  where o = number of out dims
    # nt = number ot test samples
    proj = core_res.projw * view(core_res.projLDA, :, 1:out_dim) #proj is the projection_matrix
    XWt = MMI.matrix(Xnew) * proj
    # centroids in the transformed space, nc x o
    centroids = transpose(core_res.cmeans) * proj

    # compute the distances in the transformed space between pairs of rows
    # the probability matrix is `nt x nc` and normalised accross rows
    Pr = pairwise(m.dist, XWt, centroids, dims=1)
    Pr .*= -1
    # apply a softmax transformation
    softmax!(Pr)
    return MMI.UnivariateFinite(classes_seen, Pr)
end

metadata_model(
    SubspaceLDA,
    input=Table(Continuous),
    target=AbstractVector{<:Finite},
    weights=false,
    output=Table(Continuous),
    descr=SubspaceLDA_DESCR,
    path="$(PKG).SubspaceLDA"
)

####
#### BayesianSubspaceLDA
####

"""
    BayesianSubspaceLDA(; kwargs...)

$BayesianSubspaceLDA_DESCR

# Keyword Parameters

- `normalize::Bool=true`: Option to normalize the between class variance for the number of
    observations in each class, one of `true` or `false`.
- `out_dim::Int=0`: the dimension of the transformed space to be used by `predict` and 
    `transform` methods, automatically set if 0 is given (default).
- `priors::Union{Nothing, Vector{Float64}}=nothing`: For use in prediction with Baye's 
    rule. If `priors = nothing` then `priors` are estimated from the class proportions 
    in the training data. Otherwise it requires a `Vector` containing class 
    probabilities with probabilities specified using the order given by `levels(y)` 
    where y is the target vector.

For more information about the algorithm, see the paper by Howland & Park (2006), 
"Generalizing discriminant analysis using the generalized singular value decomposition"
,IEEE Trans. Patt. Anal. & Mach. Int., 26: 995-1006.
"""
@mlj_model mutable struct BayesianSubspaceLDA <: MMI.Probabilistic
    normalize::Bool=false
    out_dim::Int= 0::(_ ≥ 0)
    priors::Union{Nothing, Vector{Float64}}=nothing
end

function MMI.fit(model::BayesianSubspaceLDA, ::Int, X, y)
    Xm_t, yplain, classes_seen, p, n, nc, nclasses, integers_seen, out_dim = 
        _check_lda_data(model, X, y) 
    ## If piors are specified check if they makes sense.
    ## This was put here to through errors much earlier
    if isa(model.priors, Vector)
        priors = _check_lda_priors(model.priors, nc, nclasses, integers_seen)         
    end
    
    core_res = MS.fit(
        MS.SubspaceLDA, Xm_t, Int.(yplain), nc;
        normalize = model.normalize
    )
    # λ is a (nc -1) x 1 vector containing the eigen values sorted in descending order.
    λ = core_res.λ
    explained_variance_ratio = λ ./ sum(λ) #proportions of variance
    mult = model.normalize ? n : 1 #used in prediction

    ## Estimates prior probabilities if specified by user.
    ## Put it here to avoid recomputing as the fitting process does this already
    if isa(model.priors, Nothing)
        weights = MS.classweights(core_res)
        priors = weights ./ n
    end

    cache = nothing
    report = (
        explained_variance_ratio=explained_variance_ratio,
        classes=classes_seen,
        class_means=MS.classmeans(core_res),
        mean=MS.mean(core_res),
        class_weights=MS.classweights(core_res),
        nc=nc
    )
    fitresult = (core_res, out_dim, classes_seen, priors, n, mult)
    return fitresult, cache, report
end

function _matrix_transpose(model::Union{SubspaceLDA, BayesianSubspaceLDA}, X)
    return transpose(MMI.matrix(X))
end
 
function MMI.fitted_params(::BayesianSubspaceLDA, (core_res, _, _, priors,_))
    return (
        projected_class_means=MS.classmeans(core_res),
        projection_matrix=MS.projection(core_res),
        priors=_get_priors(priors)
    )
end

function MMI.predict(
    m::BayesianSubspaceLDA,
    (core_res, out_dim, classes_seen, priors, n, mult),
    Xnew
)
    # projection of Xnew, XWt is nt x o  where o = number of out dims
    # nt = number ot test samples
    #proj is the projection_matrix
    proj = core_res.projw * view(core_res.projLDA, :, 1:out_dim)
    XWt = MMI.matrix(Xnew) * proj
    
    # centroids in the transformed space, nc x o
    centroids = transpose(core_res.cmeans) * proj
    nc = length(classes_seen)

    # compute the distances in the transformed space between pairs of rows
    # The discriminant matrix `Pr` is of dimension `nt x nc`
    # Pr[i,k] = -0.5*(xᵢ −  µₖ)ᵀ(Σw⁻¹)(xᵢ −  µₖ) + log(priorsₖ) where Σw = Sw/(n-nc)
    # In the transformed space this becomes
    # Pr[i,k] = -0.5*(Pᵀxᵢ −  Pᵀµₖ)ᵀ(PᵀΣw⁻¹P)(Pᵀxᵢ −  Pᵀµₖ) + log(priorsₖ)
    # But PᵀSw⁻¹P = (1/mult)*I and PᵀΣw⁻¹P = (n-nc)/mult*I
    # Giving Pr[i,k] = -0.5*n*(Pᵀxᵢ −  Pᵀµₖ)ᵀ(Pᵀxᵢ −  Pᵀµₖ) + log(priorsₖ)
    # (Pᵀxᵢ −  Pᵀµₖ)ᵀ(Pᵀxᵢ −  Pᵀµₖ) is the SquaredEquclidean distance in the 
    # transformed space  
    Pr = pairwise(SqEuclidean(), XWt, centroids, dims=1)
    Pr .*= (-(n-nc)/2mult)
    Pr .+= log.(transpose(priors))

    # apply a softmax transformation to convert Pr to a probability matrix
    softmax!(Pr)
    return MMI.UnivariateFinite(classes_seen, Pr)
end

function MMI.transform(m::T, (core_res, out_dim, _), X) where T<:Union{SubspaceLDA, BayesianSubspaceLDA}
    # projection of X, XWt is nt x o  where o = out dims
    proj = core_res.projw * view(core_res.projLDA, :, 1:out_dim) #proj is the projection_matrix
    XWt = MMI.matrix(X) * proj
    return MMI.table(XWt, prototype = X)
end

metadata_model(
    BayesianSubspaceLDA,
    input=Table(Continuous),
    target=AbstractVector{<:Finite},
    weights=false,
    output=Table(Continuous),
    descr=BayesianSubspaceLDA_DESCR,
    path="$(PKG).BayesianSubspaceLDA"
)
