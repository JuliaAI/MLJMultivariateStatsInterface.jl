@testset "replace!" begin
    y = [1, 2, 2, 2, 3]
    z1 = 1:5
    z2 = 0:3
    r1 = -2:2
    r2 = 2:5    
    @test_throws DimensionMismatch _replace!(y, z1, r2)
    @test_throws DimensionMismatch _replace!(y, z2, r1)
    @test _replace!(deepcopy(y), z1, r1) == Base.replace!(deepcopy(y), (z1 .=> r1)...)
    @test _replace!(deepcopy(y), z2, r2) == Base.replace!(deepcopy(y), (z2 .=> r2)...)
end

@testset "softmax" begin
    X = rand(100,5)
    max_ = maximum(X, dims=2)
    exp_ = exp.(X .- max_)
    @test isapprox(MLJMultivariateStatsInterface.softmax(X), exp_ ./ sum(exp_, dims=2))
    X_ = MLJMultivariateStatsInterface.softmax!(X)
    @test X_ === X
    @test isapprox(X_, exp_ ./ sum(exp_, dims=2))     
end

