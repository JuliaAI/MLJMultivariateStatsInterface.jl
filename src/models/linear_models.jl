#######
## Common Regressor methods
########

struct LinearFitresult{T, F<:Real, M<:AbstractArray{F}} <: MMI.MLJType
    sol_matrix::M
    bias::Bool
    names::T
end

_convert(common_type, x::AbstractVector) = convert(AbstractVector{common_type}, x)
_convert(common_type, x::AbstractMatrix) = convert(AbstractMatrix{common_type}, MMI.matrix(x))
matrix_(X::AbstractVector) = X
matrix_(X) = MMI.matrix(X)
_names(y::AbstractVector) = nothing
_names(Y) = collect(MMI.schema(Y).names)

function _matrix(X, target)
    Xmatrix_ = MMI.matrix(X)
    Y_ = matrix_(target)
    common_type = promote_type(eltype(Xmatrix_), eltype(Y_))
    Xmatrix = _convert(common_type, Xmatrix_)
    Y = _convert(common_type, Y_)
    return Xmatrix, Y, _names(target)
end

function _regressor_fitted_params(fr::LinearFitresult{Nothing, <:Real, <:AbstractVector})
    return (
        coefficients=fr.sol_matrix[1:end-Int(fr.bias)],
        intercept=ifelse(fr.bias, fr.sol_matrix[end], nothing)
    )
end

function _regressor_fitted_params(fr::LinearFitresult{<:Vector, <:Real, <:AbstractMatrix})
    return (
        coefficients=fr.sol_matrix[1:end-Int(fr.bias), :],
        intercept=fr.bias ? fr.sol_matrix[end, :] : nothing
    )
end

function _predict_regressor(
    fr::LinearFitresult{Nothing, <:Real, <:AbstractVector},
    Xmat_new::AbstractMatrix,
    prototype
)
    if fr.bias
        return @views Xmat_new * fr.sol_matrix[1:end-1] .+ transpose(fr.sol_matrix[end])
    else
        return @views Xmat_new * fr.sol_matrix[1:end-1]
    end
end

function _predict_regressor(
    fr::LinearFitresult{<:Vector, <:Real, <:AbstractMatrix},
    Xmat_new::AbstractMatrix,
    prototype
)
    if fr.bias
        return MMI.table(
            Xmat_new * @view(fr.sol_matrix[1:end-1, :]) .+ transpose(
                @view(fr.sol_matrix[end, :])
            );
            names=fr.names,
            prototype=prototype
        )
    else
        return MMI.table(
            Xmat_new * @view(fr.sol_matrix[1:end-1, :]);
            names=fr.names,
            prototype=prototype
        )
    end
end

####
#### LinearRegressor & MultitargetLinearRegressor
####

@mlj_model mutable struct LinearRegressor <: MMI.Deterministic
    bias::Bool = true
end

@mlj_model mutable struct MultitargetLinearRegressor <: MMI.Deterministic
    bias::Bool = true
end

const LINREG = Union{LinearRegressor, MultitargetLinearRegressor}

function MMI.fit(model::LINREG, verbosity::Int, X, y)
    Xmatrix, y_, target_header= _matrix(X, y)
    θ = MS.llsq(Xmatrix, y_; bias=model.bias)
    fitresult = LinearFitresult(θ, model.bias, target_header)
    report = NamedTuple()
    cache = nothing
    return fitresult, cache, report
end

function MMI.fitted_params(::LINREG, fr)
    return _regressor_fitted_params(fr)
end

function MMI.predict(::LINREG, fr, Xnew)
    Xmat_new = MMI.matrix(Xnew)
    return _predict_regressor(fr, Xmat_new, Xnew)
end

####
#### RidgeRegressor & MultitargetRidgeRegressor
####

_check_typeof_lambda(x)= x isa AbstractVecOrMat || (x isa Real && x ≥ 0)

@mlj_model mutable struct RidgeRegressor <: MMI.Deterministic
    lambda::Union{Real, AbstractVecOrMat} = 1.0::(_check_typeof_lambda(_))
    bias::Bool = true
end

@mlj_model mutable struct MultitargetRidgeRegressor <: MMI.Deterministic
    lambda::Union{Real, AbstractVecOrMat} = 1.0::(_check_typeof_lambda(_))
    bias::Bool = true
end

const RIDGEREG = Union{RidgeRegressor, MultitargetRidgeRegressor}

function MMI.fit(model::RIDGEREG, verbosity::Int, X, y)
    Xmatrix, y_, target_header = _matrix(X, y)
    θ = MS.ridge(Xmatrix, y_, model.lambda; bias=model.bias)
    fitresult = LinearFitresult(θ, model.bias, target_header)
    report = NamedTuple()
    cache = nothing
    return fitresult, cache, report
end

function MMI.fitted_params(::RIDGEREG, fr)
    return _regressor_fitted_params(fr)
end

function MMI.predict(::RIDGEREG, fr, Xnew)
    Xmat_new = MMI.matrix(Xnew)
    return _predict_regressor(fr, Xmat_new, Xnew)
end


############
### Models Metadata
############

metadata_model(
    LinearRegressor,
    input=Table(Continuous),
    target=AbstractVector{Continuous},
    weights=false,
    path="$(PKG).LinearRegressor"
)

metadata_model(
    MultitargetLinearRegressor,
    input=Table(Continuous),
    target=Table(Continuous),
    weights=false,
    path="$(PKG).MultitargetLinearRegressor"
)

metadata_model(
    RidgeRegressor,
    input=Table(Continuous),
    target=AbstractVector{Continuous},
    weights=false,
    path="$(PKG).RidgeRegressor"
)

metadata_model(
    MultitargetRidgeRegressor,
    input=Table(Continuous),
    target=Table(Continuous),
    weights=false,
    path="$(PKG).MultitargetRidgeRegressor"
)
