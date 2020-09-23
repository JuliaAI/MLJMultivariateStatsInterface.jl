####
#### LinearRegressor
####

"""
    LinearRegressor(; bias::Bool=true)

$LINEAR_DESCR

# Keyword Parameters

- `bias::Bool=true`: if true includes a bias term else fits without bias term.
"""
@mlj_model mutable struct LinearRegressor <: MMI.Deterministic
    bias::Bool = true
end

struct LinearFitresult{F<:Real, M<:AbstractArray{F}} <: MMI.MLJType
    sol_matrix::M
    bias::Bool
end

_convert(common_type, x::AbstractVector) = convert(AbstractVector{common_type}, x)
_convert(common_type, x::AbstractMatrix) = convert(AbstractMatrix{common_type}, MMI.matrix(x))
matrix_(X::AbstractVector) = X
matrix_(X) = MMI.matrix(X) 

function _matrix(X, target)
    Xmatrix_ = MMI.matrix(X)
    Y_ = matrix_(target)
    common_type = promote_type(eltype(Xmatrix_), eltype(Y_))
    Xmatrix = _convert(common_type, Xmatrix_)
    Y = _convert(common_type, Y_)
    return Xmatrix, Y
end

function MMI.fit(model::LinearRegressor, verbosity::Int, X, y)
    Xmatrix, y = _matrix(X, y)
    θ = MS.llsq(Xmatrix, y; bias=model.bias)
    fitresult = LinearFitresult(θ, model.bias)
    report = NamedTuple()
    cache = nothing
    return fitresult, cache, report
end

function _regressor_fitted_params(fr::LinearFitresult{<:Real, <:AbstractVector})
    return (
        coefficients=fr.sol_matrix[1:end-Int(fr.bias)],
        intercept=ifelse(fr.bias, fr.sol_matrix[end], nothing)
    )
end

function _regressor_fitted_params(fr::LinearFitresult{<:Real, <:AbstractMatrix})
    return (
        coefficients=fr.sol_matrix[1:end-Int(fr.bias), :],
        intercept=ifelse(fr.bias, fr.sol_matrix[end, :], nothing)
    )
end

function MMI.fitted_params(::LinearRegressor, fr)
    return _regressor_fitted_params(fr)
end

function _predict_regressor(fr::LinearFitresult{<:Real, <:AbstractVector}, Xmat_new)
    if fr.bias
        return @views Xmat_new * fr.sol_matrix[1:end-1] .+ transpose(fr.sol_matrix[end])
    else
        return @views Xmat_new * fr.sol_matrix[1:end-1]
    end
end

function _predict_regressor(fr::LinearFitresult{<:Real, <:AbstractMatrix}, Xmat_new)
    if fr.bias
        return MMI.table(
            Xmat_new * @view(fr.sol_matrix[1:end-1, :]) .+ transpose(
                @view(fr.sol_matrix[end, :])
            ),
            prototype=Xmat_new
        )
    else
        return MMI.table(Xmat_new * @view(fr.sol_matrix[1:end-1, :]), prototype=Xmat_new)
    end
end

function MMI.predict(::LinearRegressor, fr, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    return _predict_regressor(fr, Xmatrix)
end

metadata_model(
    LinearRegressor,
    input=Table(Continuous),
    target=Union{Table(Continuous), AbstractVector{Continuous}},
    weights=false,
    descr=LINEAR_DESCR,
    path="$(PKG).LinearRegressor"
)

####
#### RidgeRegressor
####

_check_typeof_lambda(x)= x isa AbstractVecOrMat || (x isa Real && x ≥ 0)

"""
    RidgeRegressor(; lambda::Union{Real, AbstractVecOrMat}=1.0, bias::Bool=true)

$RIDGE_DESCR

# Keyword Parameters

- `lambda::Union{Real, AbstractVecOrMat}=1.0`: non-negative parameter for the 
    regularization strength.
- `bias::Bool=true`: if true includes a bias term else fits without bias term.
"""
@mlj_model mutable struct RidgeRegressor <: MMI.Deterministic
    lambda::Union{Real, AbstractVecOrMat} = 1.0::(_check_typeof_lambda(_))
    bias::Bool = true
end

function MMI.fit(model::RidgeRegressor, verbosity::Int, X, y)
    Xmatrix, y = _matrix(X, y)
    θ = MS.ridge(Xmatrix, y, model.lambda; bias=model.bias)
    fitresult = LinearFitresult(θ, model.bias)
    report = NamedTuple()
    cache = nothing
    return fitresult, cache, report
end

function MMI.fitted_params(::RidgeRegressor, fr)
    return _regressor_fitted_params(fr)
end

function MMI.predict(::RidgeRegressor, fr, Xnew)
    Xmatrix = MMI.matrix(Xnew)
    return _predict_regressor(fr, Xmatrix)
end

metadata_model(
    RidgeRegressor,
    input=Table(Continuous),
    target=Union{Table(Continuous), AbstractVector{Continuous}},
    weights=false,
    descr=RIDGE_DESCR,
    path="$(PKG).RidgeRegressor"
)
