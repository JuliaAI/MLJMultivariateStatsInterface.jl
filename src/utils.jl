# internal method essentially the same as Base.replace!(y, (z .=> r)...)
# but more efficient.
# Similar to the behaviour of `Base.replace!` if `z` contain repetions of values in 
# `y` then only the transformation corresponding to the first occurence is performed
# i.e `_replace!([1,5,3], [1,4], 4:5)` would return `[4,5,3]` rather than `[5,5,3]`
# (which replaces `1=>4` and then `4=>5`)
function _replace!(y::AbstractVector, z::AbstractVector, r::AbstractVector)
    length(r) == length(z) || 
     throw(DimensionMismatch("`z` and `r` has to be of the same length"))
    @inbounds for i in eachindex(y)
        for j in eachindex(z) 
            isequal(z[j], y[i]) && (y[i] = r[j]; break)
        end
    end
    return y
end

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
