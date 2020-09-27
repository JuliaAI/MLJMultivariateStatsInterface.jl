"""
 softmax(X::AbstractMatrix{<:Real})
Return the softmax computed in a numerically stable way:
``σ(x) = exp.(x) ./ sum(exp.(x))``
Implementation taken from NNlib.jl.
"""
function softmax(X::AbstractMatrix{<:Real})
    S = copyto!(similar(X, eltype(X)), X)
    return softmax!(S)
end

"""
 softmax!(X::AbstractMatrix{<:Real})
Return the softmax computed in a numerically stable way:
``σ(x) = exp.(x) ./ sum(exp.(x))`` and store the result in the
input matrix
Implementation taken from NNlib.jl.
"""
function softmax!(X::AbstractMatrix{<:Real})
    max_ = maximum(X, dims=2)
    X .= exp.(X .- max_) 
    X ./= sum(X, dims=2)
    return X 
end
