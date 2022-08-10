####
#### MulticlassLDA
####

@mlj_model mutable struct LDA <: MMI.Probabilistic
    method::Symbol = :gevd::(_ in (:gevd, :whiten))
    cov_w::CovarianceEstimator = MS.SimpleCovariance()
    cov_b::CovarianceEstimator = MS.SimpleCovariance()
    outdim::Int = 0::(_ ≥ 0)
    regcoef::Float64 = 1e-6::(_ ≥ 0)
    dist::SemiMetric = SqEuclidean()
end

function MMI.fit(model::LDA, ::Int, X, y)
    Xm_t, yplain, classes_seen, p, n, nc, nclasses, integers_seen, outdim =
        _check_lda_data(model, X, y)
    core_res = MS.fit(
        MS.MulticlassLDA, nc, Xm_t, Int.(yplain);
        method=model.method,
        outdim,
        regcoef=model.regcoef,
        covestimator_within=model.cov_w,
        covestimator_between=model.cov_b
    )
    cache = nothing
    report = (
        classes=classes_seen,
        outdim=MS.size(core_res)[2],
        projected_class_means=MS.classmeans(core_res),
        mean=MS.mean(core_res),
        class_weights=MS.classweights(core_res),
        Sw=MS.withclass_scatter(core_res),
        Sb=MS.betweenclass_scatter(core_res),
        nclasses=nc
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
    outdim = ifelse(model.outdim == 0, def_outdim, model.outdim)
    # Check if the given one is sensible
    if outdim > p
        throw(
            ArgumentError(
                "`outdim` must not be larger than `p`"*
                "where `p` is the number of features in `X`"
            )
        )
    end
    return Xm_t, yplain, classes_seen, p, n, nc, nclasses, integers_seen, outdim
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

metadata_model(
    LDA,
    human_name="linear discriminant analysis model",
    input=Table(Continuous),
    target=AbstractVector{<:Finite},
    weights=false,
    output=Table(Continuous),
    path="$(PKG).LDA"
)


####
#### BayesianLDA
####

@mlj_model mutable struct BayesianLDA <: MMI.Probabilistic
    method::Symbol = :gevd::(_ in (:gevd, :whiten))
    cov_w::CovarianceEstimator=MS.SimpleCovariance()
    cov_b::CovarianceEstimator=MS.SimpleCovariance()
    outdim::Int=0::(_ ≥ 0)
    regcoef::Float64=1e-6::(_ ≥ 0)
    priors::Union{Nothing, Vector{Float64}}=nothing
end

function MMI.fit(model::BayesianLDA, ::Int, X, y)
    Xm_t, yplain, classes_seen, p, n, nc, nclasses, integers_seen, outdim =
        _check_lda_data(model, X, y)
    ## If piors are specified check if they makes sense.
    ## This was put here to through errors much earlier
    if isa(model.priors, Vector)
        priors = _check_lda_priors(model.priors, nc, nclasses, integers_seen)
    end

    core_res = MS.fit(
        MS.MulticlassLDA, nc, Xm_t, Int.(yplain);
        method=model.method,
        outdim,
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
        outdim=MS.size(core_res)[2],
        projected_class_means=MS.classmeans(core_res),
        mean=MS.mean(core_res),
        class_weights=MS.classweights(core_res),
        Sw=MS.withclass_scatter(core_res),
        Sb=MS.betweenclass_scatter(core_res),
        nclasses=nc
    )

    fitresult = (core_res, classes_seen, priors, n)
    return fitresult, cache, report
end

function _matrix_transpose(::Union{LDA,BayesianLDA}, X)
    # MultivariateStats 9.0 is not supporting adjoints
    return MMI.matrix(X, transpose=true)
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
    human_name="Bayesian LDA model",
    input=Table(Continuous),
    target= AbstractVector{<:Finite},
    weights=false,
    output=Table(Continuous),
    path="$(PKG).BayesianLDA"
)

####
#### SubspaceLDA
####

@mlj_model mutable struct SubspaceLDA <: MMI.Probabilistic
    normalize::Bool = true
    outdim::Int = 0::(_ ≥ 0)
    dist::SemiMetric = SqEuclidean()
end

function MMI.fit(model::SubspaceLDA, ::Int, X, y)
    Xm_t, yplain, classes_seen, p, n, nc, nclasses, integers_seen, outdim =
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
        projected_class_means=MS.classmeans(core_res),
        mean=MS.mean(core_res),
        class_weights=MS.classweights(core_res),
        nclasses=nc,
    )
    fitresult = (core_res, outdim, classes_seen)
    return fitresult, cache, report
end

function MMI.fitted_params(::SubspaceLDA, (core_res, _))
    return (projected_class_means=MS.classmeans(core_res), projection_matrix=MS.projection(core_res))
end

function MMI.predict(m::SubspaceLDA, (core_res, outdim, classes_seen), Xnew)
     # projection of Xnew, XWt is nt x o  where o = number of out dims
    # nt = number ot test samples
    proj = core_res.projw * view(core_res.projLDA, :, 1:outdim) #proj is the projection_matrix
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
    human_name="subpace LDA model",
    input=Table(Continuous),
    target=AbstractVector{<:Finite},
    weights=false,
    output=Table(Continuous),
    path="$(PKG).SubspaceLDA"
)

####
#### BayesianSubspaceLDA
####

@mlj_model mutable struct BayesianSubspaceLDA <: MMI.Probabilistic
    normalize::Bool=false
    outdim::Int= 0::(_ ≥ 0)
    priors::Union{Nothing, Vector{Float64}}=nothing
end

function MMI.fit(model::BayesianSubspaceLDA, ::Int, X, y)
    Xm_t, yplain, classes_seen, p, n, nc, nclasses, integers_seen, outdim =
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
        projected_class_means=MS.classmeans(core_res),
        mean=MS.mean(core_res),
        class_weights=MS.classweights(core_res),
        nclasses=nc
    )
    fitresult = (core_res, outdim, classes_seen, priors, n, mult)
    return fitresult, cache, report
end

function _matrix_transpose(model::Union{SubspaceLDA, BayesianSubspaceLDA}, X)
    return MMI.matrix(X)'
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
    (core_res, outdim, classes_seen, priors, n, mult),
    Xnew
)
    # projection of Xnew, XWt is nt x o  where o = number of out dims
    # nt = number ot test samples
    #proj is the projection_matrix
    proj = core_res.projw * view(core_res.projLDA, :, 1:outdim)
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

function MMI.transform(m::T, (core_res, outdim, _), X) where T<:Union{SubspaceLDA, BayesianSubspaceLDA}
    # projection of X, XWt is nt x o  where o = out dims
    proj = core_res.projw * view(core_res.projLDA, :, 1:outdim)
    #proj is the projection_matrix
    XWt = MMI.matrix(X) * proj
    return MMI.table(XWt, prototype = X)
end

metadata_model(
    BayesianSubspaceLDA,
    human_name="Bayesian subspace LDA model",
    input=Table(Continuous),
    target=AbstractVector{<:Finite},
    weights=false,
    output=Table(Continuous),
    path="$(PKG).BayesianSubspaceLDA"
)


# # DOCUMENT STRINGS

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
