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

metadata_pkg.(
    [
        LinearRegressor,
        MultitargetLinearRegressor,
        RidgeRegressor,
        MultitargetRidgeRegressor,
    ],
    name = "MultivariateStats",
    uuid = "6f286f6a-111f-5878-ab1e-185364afe411",
    url = "https://github.com/JuliaStats/MultivariateStats.jl",
    license = "MIT",
    julia = true,
    is_wrapper = false
)


# # DOCUMENT STRINGS

"""

$(MMI.doc_header(LinearRegressor))

`LinearRegressor` assumes the target is a `Continuous` variable and trains a linear
prediction function using the least squares algorithm. Options exist to specify a bias 
term.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype 
    `Continuous`; check the column scitypes with `schema(X)`.

- `y` is the target, which can be any `AbstractVector` whose element scitype is 
    `Continuous`; check the scitype with `scitype(y)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new features `Xnew`, which 
    should have the same scitype as `X` above.

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

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype 
    `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any table of responses whose element scitype is 
    `Continuous`; check the scitype with `scitype(y)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new features `Xnew`, 
    which should have the same scitype as `X` above.

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

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype 
    `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any `AbstractVector` whose element scitype is 
    `Continuous`; check the scitype with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `lambda=1.0`: Is the non-negative parameter for the regularization strength. If lambda 
    is 0, ridge regression is equivalent to linear least squares regression, and as lambda 
    approaches infinity, all the linear coefficients approach 0.

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new features `Xnew`, which 
    should have the same scitype as `X` above.

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

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype 
    `Continuous`; check column scitypes with `schema(X)`.

- `y` is the target, which can be any table of responses whose element scitype is 
    `Continuous`; check the scitype with `scitype(y)`.

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `lambda=1.0`: Is the non-negative parameter for the regularization strength. If lambda 
    is 0, ridge regression is equivalent to linear least squares regression, and as lambda 
    approaches infinity, all the linear coefficients approach 0.

- `bias=true`: Include the bias term if true, otherwise fit without bias term.

# Operations

- `predict(mach, Xnew)`: Return predictions of the target given new features `Xnew`, which 
    should have the same scitype as `X` above.

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
