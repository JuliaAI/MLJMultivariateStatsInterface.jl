@testset "Single-response Linear" begin
    # Define some linear, noise-free, synthetic data:
    # with intercept = 0.0
    n = 1000
    rng = StableRNG(1234)
    X, y = MLJBase.make_regression(n, 3, noise=0, intercept=false, as_table=false, rng=rng)
    # Train model with intercept on all data
    linear = LinearRegressor()
    yhat, fr = test_regression(linear, X, y)
    # Training error
    @test norm(yhat - y)/sqrt(n) < 1e-12
    # Get the true intercept?
    @test abs(fr.intercept) < 1e-10
    d = info_dict(linear)
    @test d[:input_scitype] == Table(Continuous)
    @test d[:target_scitype] == Union{Table(Continuous), AbstractVector{Continuous}}
    @test d[:name] == "LinearRegressor"
end

@testset "Multi-response Linear" begin
    # Define some linear, noise-free, synthetic data:
    # with intercept = 0.0
    n = 1000
    rng = StableRNG(1234)
    X, Y = make_regression2(n, 3, noise=0, intercept=false, rng=rng)
    # Train model on all data
    linear = LinearRegressor()
    Yhat_, fr = test_regression(linear, X, Y)
    Yhat = MLJBase.matrix(Yhat_)
    # Training error
    @test norm(Yhat - Y)/sqrt(n) < 1e-12
    # Get the true intercept?
    @test norm(fr.intercept .- zeros(size(Y, 2))) < 1e-10
end

@testset "Single-response Ridge" begin
    # Define some linear, noise-free, synthetic data:
    # with intercept = 0.0
    n = 1000
    rng = StableRNG(1234)
    X, y = MLJBase.make_regression(n, 3, noise=0, intercept=false, as_table=false, rng=rng)
    # Train model with intercept on all data with no regularization
    # and no standardization of target.
    ridge = RidgeRegressor(lambda=0.0)
    yhat, fr = test_regression(ridge, X, y)
    # Training error
    @test norm(yhat - y)/sqrt(n) < 1e-12
    # Get the true intercept?
    @test abs(fr.intercept) < 1e-10
    d = info_dict(ridge)
    @test d[:input_scitype] == Table(Continuous)
    @test d[:target_scitype] == Union{Table(Continuous), AbstractVector{Continuous}}
    @test d[:name] == "RidgeRegressor"
end

@testset "Multi-response Ridge" begin
    # Define some linear, noise-free, synthetic data:
    # with intercept = 0.0
    n = 1000
    rng = StableRNG(1234)
    X, Y = make_regression2(n, 3, noise=0, intercept=false, rng=rng)
    # Train model with intercept on all data with no regularization 
    # and no standardization of target.
    ridge = RidgeRegressor(lambda=0.0)
    Yhat_, fr = test_regression(ridge, X, Y)
    Yhat = MLJBase.matrix(Yhat_)
    # Training error
    @test norm(Yhat - Y)/sqrt(n) < 1e-12
    # Get the true intercept?
    @test norm(fr.intercept .- zeros(size(Y, 2))) < 1e-10
end
