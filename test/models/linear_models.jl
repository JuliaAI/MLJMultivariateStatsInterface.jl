@testset "LinearRegressor" begin
    # Define some linear, noise-free, synthetic data:
    # with intercept = 0.0
    n = 1000
    rng = StableRNG(1234)
    X, y = MLJBase.make_regression(n, 3, noise=0, intercept=false, rng=rng)
    # Train model with intercept on all data
    linear = LinearRegressor()
    yhat, fr = test_regression(linear, X, y)
    # Training error
    @test norm(yhat - y)/sqrt(n) < 1e-12
    # Get the true intercept?
    @test abs(fr.intercept) < 1e-10
    # Check metadata
end

@testset "MultitargetLinearRegressor" begin
    # Define some linear, noise-free, synthetic data:
    # with intercept = 0.0
    n = 1000
    rng = StableRNG(1234)
    X, Y = make_regression2(n, 3, noise=0, intercept=false, rng=rng)
    Y_mat = MLJBase.matrix(Y)
    # Train model on all data
    multi_linear = MultitargetLinearRegressor()
    Yhat, fr = test_regression(multi_linear, X, Y)
    Yhat_mat = MLJBase.matrix(Yhat)
    # Check if the column names is same after predict
    MLJBase.schema(Yhat).names == MLJBase.schema(Y).names
    # Training error
    @test norm(Yhat_mat - Y_mat)/sqrt(n) < 1e-12
    # Get the true intercept?
    @test norm(fr.intercept .- zeros(size(Y_mat, 2))) < 1e-10
end

@testset "RidgeRegressor" begin
    # Define some linear, noise-free, synthetic data:
    # with intercept = 0.0
    n = 1000
    rng = StableRNG(1234)
    X, y = MLJBase.make_regression(n, 3, noise=0, intercept=false, rng=rng)
    # Train model with intercept on all data with no regularization
    # and no standardization of target.
    ridge = RidgeRegressor(lambda=0.0)
    yhat, fr = test_regression(ridge, X, y)
    # Training error
    @test norm(yhat - y)/sqrt(n) < 1e-12
    # Get the true intercept?
    @test abs(fr.intercept) < 1e-10
    # Check metadata
end

@testset "MultitargetRidgeRegressor" begin
    # Define some linear, noise-free, synthetic data:
    # with intercept = 0.0
    n = 1000
    rng = StableRNG(1234)
    X, Y = make_regression2(n, 3, noise=0, intercept=false, rng=rng)
    Y_mat = MLJBase.matrix(Y)
    # Train model with intercept on all data with no regularization
    # and no standardization of target.
    multi_ridge = MultitargetRidgeRegressor(lambda=0.0)
    Yhat, fr = test_regression(multi_ridge, X, Y)
    Yhat_mat = MLJBase.matrix(Yhat)
    # Check if the column names is same after predict
    MLJBase.schema(Yhat).names == MLJBase.schema(Y).names
    # Training error
    @test norm(Yhat_mat - Y_mat)/sqrt(n) < 1e-12
    # Get the true intercept?
    @test norm(fr.intercept .- zeros(size(Y_mat, 2))) < 1e-10
end
