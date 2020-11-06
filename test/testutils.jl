function make_regression2(n=100, p=3, c=5;intercept=true, noise=0.1, rng=nothing)
    if rng === nothing
        rng = Random.GLOBAL_RNG
    elseif rng isa Integer
        rng = Random.MersenneTwister(rng)
    end
    
    X = randn(rng, n, p)
    X = MLJBase.augment_X(randn(rng, n, p), intercept)
    A = randn(rng, p + Int(intercept), c)
    Y = (X * A) .+ (noise .* randn(rng, n, c))
    return MLJBase.table(X), MLJBase.table(Y, names=[Symbol("y$(i)") for i = 1:c])
end

function test_regression(model, X, y)
    fitresult, report, cache = fit(model, 0, X, y)
    yhat = predict(model, fitresult, X)
    fr = fitted_params(model, fitresult)
    return yhat, fr
end

function test_composition_model(ms_model, mlj_model, X, X_array ; test_inverse=true)
    mlj_model_type = typeof(mlj_model)
    Xtr_ms = permutedims(
        MultivariateStats.transform(ms_model, permutedims(X_array))
    )  
    fitresult, _, _ = fit(mlj_model, 1, X)
    Xtr_mlj_table = transform(mlj_model, fitresult, X)
    Xtr_mlj = matrix(Xtr_mlj_table)
    # Compare MLJ and MultivariateStats transformed matrices
    @test Xtr_mlj ≈ Xtr_ms
    # test metadata
    d = info_dict(mlj_model_type)
    @test d[:input_scitype] == Table(Continuous)
    @test d[:output_scitype] == Table(Continuous)
    @test d[:name] == string(mlj_model_type)

    if test_inverse
        Xinv_ms = permutedims(
            MultivariateStats.reconstruct(ms_model, permutedims(Xtr_ms))
        )
        Xinv_mlj_table = inverse_transform(mlj_model, fitresult, Xtr_mlj)
        Xinv_mlj = matrix(Xinv_mlj_table)
        @test Xinv_ms ≈ Xinv_mlj
    end
end
