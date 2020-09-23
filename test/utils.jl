@testset "softmax" begin
    X = rand(100,5)
    max_ = maximum(X, dims=2)
    exp_ = exp.(X .- max_)
    @test isapprox(MLJMultivariateStatsInterface.softmax(X), exp_ ./ sum(exp_, dims=2))
    X_ = MLJMultivariateStatsInterface.softmax!(X)
    @test X_ === X
    @test isapprox(X_, exp_ ./ sum(exp_, dims=2))     
end

