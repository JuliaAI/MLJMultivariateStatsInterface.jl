@testset "MulticlassLDA" begin
    ## Data
    Xfull, y = @load_smarket
    X = selectcols(Xfull, [:Lag1,:Lag2])
    train = selectcols(Xfull, :Year) .< Dates.Date(2005)
    test = .!train
    Xtrain = selectrows(X, train)
    ytrain = selectrows(y, train)
    Xtest = selectrows(X, test)
    ytest = selectrows(y, test)

    lda_model = LDA()

    ## Check model `fit`
    fitresult, _, report= fit(lda_model, 1, Xtrain, ytrain)
    classes_seen, projection_matrix = fitted_params(lda_model, fitresult)
    @test classes_seen == ["Up", "Down"]
    @test round.((report.class_means)', sigdigits = 3) == [-0.0395 -0.0313; 0.0428 0.0339] #[0.0428 0.0339; -0.0395 -0.0313]
    ## Check model `predict`
    preds = predict(lda_model, fitresult, Xtest)
    mce = cross_entropy(preds, ytest) |> mean
    @test 0.685 ≤ mce ≤ 0.695
    ## Check model `transform`
    # MultivariateStats Linear Discriminant Analysis transform
    proj = fitresult[1].proj
    XWt = matrix(X) * proj
    tlda_ms = table(XWt, prototype=X)
    # MLJ Linear Discriminant Analysis transform
    tlda_mlj = transform(lda_model, fitresult, X)
    @test tlda_mlj == tlda_ms
    ## Check model traits
end

@testset "MLDA-2" begin
    ## Data
    Random.seed!(1125)
    X1 = -2 .+ randn(100, 2)
    X2 = randn(100, 2)
    X3 = 2 .+ randn(100, 2)
    y1 = ones(100)
    y2 = 2ones(100)
    y3 = 3ones(100)
    X = vcat(X1, X2, X3)
    y = vcat(y1, y2, y3)
    p = Random.randperm(300)
    X = X[p, :]
    y = y[p]
    X = table(X)
    y = categorical(y)
    train, test = partition(eachindex(y), 0.7)
    Xtrain = selectrows(X, train)
    ytrain = selectrows(y, train)
    Xtest = selectrows(X, test)
    ytest = selectrows(y, test)

    lda_model = LDA()
    ## Check model `fit`/`predict`
    fitresult, = fit(lda_model, 1, Xtrain, ytrain)
    preds = predict_mode(lda_model, fitresult, Xtest)
    mcr = misclassification_rate(preds, ytest)
    @test mcr ≤ 0.15
end

@testset "BayesianMulticlassLDA" begin
    ## Data
    Xfull, y = @load_smarket
    X = selectcols(Xfull, [:Lag1,:Lag2])
    train = selectcols(Xfull, :Year) .< Dates.Date(2005)
    test = .!train
    Xtrain = selectrows(X, train)
    ytrain = selectrows(y, train)
    Xtest = selectrows(X, test)
    ytest = selectrows(y, test)

    BLDA_model = BayesianLDA(regcoef=0)
    
    ## Check model `fit`
    fitresult, cache, report = fit(BLDA_model, 1, Xtrain, ytrain)
    classes_seen, projection_matrix, priors = fitted_params(BLDA_model, fitresult)
    @test classes(priors) == classes(y)
    @test pdf.(priors, support(priors)) == [491/998, 507/998]
    @test classes_seen == ["Up", "Down"]
    @test round.((report.class_means)', sigdigits = 3) == [-0.0395 -0.0313; 0.0428 0.0339] #[0.0428 0.0339; -0.0395 -0.0313]
    
    ## Check model `predict`
    preds = predict(BLDA_model, fitresult, Xtest)
    mce = cross_entropy(preds, ytest) |> mean
    acr = accuracy(mode.(preds), ytest)
    @test 0.685 ≤ mce ≤ 0.695
    @test round(acr, sigdigits=2) == 0.56
    ## Check model gives consistent results when priors are assigned.
    # Either as `UnivariateFinite` or `Dict`
    BLDA_model1 = BayesianLDA(regcoef=0, priors=Dict("Up"=> 507/998, "Down"=> 491/998))
    fitresult1, cache1, report1 = fit(BLDA_model1, 1, Xtrain, ytrain)
    classes_seen1, projection_matrix1, priors1 = fitted_params(BLDA_model1, fitresult1)
    BLDA_model2 = BayesianLDA(
        regcoef=0, priors=UnivariateFinite(classes(ytrain), [491/998, 507/998])
    )
    fitresult2, cache2, report2 = fit(BLDA_model2, 1, Xtrain, ytrain)
    classes_seen2, projection_matrix2, priors2 = fitted_params(BLDA_model2, fitresult2)
    @test cache1 == cache2 == cache
    @test report1 == report2 == report
    @test classes_seen1 == classes_seen2 == classes_seen
    @test projection_matrix1 == projection_matrix2 == projection_matrix
    @test isapprox(priors, priors1)
    @test isapprox(priors, priors2)
end

@testset "BayesianSubspaceLDA" begin
    ## Data
    X, y = @load_iris
    LDA_model = BayesianSubspaceLDA()
    ## Check model `fit`
    fitresult, cache, report = fit(LDA_model, 1, X, y)
    classes_seen, projection_matrix, priors = fitted_params(
        LDA_model, fitresult
    )
    @test classes_seen == ["setosa", "versicolor", "virginica"]
    @test mean(
        abs.(
            report.class_means' - [
                5.006 3.428 1.462 0.246;
                5.936 2.770 4.260 1.326;
                6.588 2.974 5.552 2.026
            ]
        )
    ) < 0.01
    @test mean(
        abs.(
            projection_matrix ≈ [
                0.0675721   0.00271023;
                            0.127666    0.177718;
                                -0.180211   -0.0767255;
                -0.235382    0.231435
            ]
        )
    ) < 0.05
    @test levels(priors) == levels(y)
    @test pdf.(priors, support(priors)) == [1/3, 1/3, 1/3]
    @test round.(report.explained_variance_ratio, digits=4) == [0.9915, 0.0085]

    ## Check model `predict`
    preds=predict(LDA_model, fitresult, X)
    predicted_class = predict_mode(LDA_model, fitresult, X)
    mcr = misclassification_rate(predicted_class, y)
    mce = cross_entropy(preds, y) |> mean
    @test round.(mcr, sigdigits=1) == 0.02
    @test 0.04 ≤ mce ≤ 0.045
    ## Check model gives consistent results when priors are assigned.
    # Either as `UnivariateFinite` or `Dict`
    LDA_model1 = BayesianSubspaceLDA(
        priors=Dict("setosa"=> 1/3, "versicolor" => 1/3, "virginica" => 1/3)
    )
    fitresult1, cache1, report1 = fit(LDA_model1, 1, X, y)
    classes_seen1, projection_matrix1, priors1 = fitted_params(
        LDA_model1, fitresult1
    )
    LDA_model2 = BayesianSubspaceLDA(
        priors=UnivariateFinite(classes(y), [1/3, 1/3, 1/3])
    )
    fitresult2, cache2, report2 = fit(LDA_model2, 1, X, y)
    classes_seen2, projection_matrix2, priors2 = fitted_params(
        LDA_model2, fitresult2
    )
    @test cache1 == cache2 == cache
    @test report1 == report2 == report
    @test classes_seen1 == classes_seen2 == classes_seen
    @test projection_matrix1 == projection_matrix2 == projection_matrix
    @test isapprox(priors, priors1)
    @test isapprox(priors, priors2)
end

@testset "SubspaceLDA" begin
    ## Data
    Random.seed!(1125)
    X1 = -2 .+ randn(100, 2)
    X2 = randn(100, 2)
    X3 = 2 .+ randn(100, 2)
    y1 = ones(100)
    y2 = 2ones(100)
    y3 = 3ones(100)
    X = vcat(X1, X2, X3)
    y = vcat(y1, y2, y3)
    p = Random.randperm(300)
    X = X[p, :]
    y = y[p]
    X = table(X)
    y = categorical(y)
    train, test = partition(eachindex(y), 0.7)
    Xtrain = selectrows(X, train)
    ytrain = selectrows(y, train)
    Xtest = selectrows(X, test)
    ytest = selectrows(y, test)

    lda_model = SubspaceLDA()
    ## Check model `fit`/ `transform`
    fitresult, = fit(lda_model, 1, Xtrain, ytrain)
    preds = predict_mode(lda_model, fitresult, Xtest)
    mcr = misclassification_rate(preds, ytest)
    @test mcr ≤ 0.15
    # MultivariateStats Linear Discriminant Analysis transform
    proj = fitresult[1].projw
    projLDA = fitresult[1].projLDA
    proj = proj * projLDA
    XWt = matrix(X) * proj
    tlda_ms = table(XWt, prototype=X)
    # MLJ Linear Discriminant Analysis transform
    tlda_mlj = transform(lda_model, fitresult, X)
    @test tlda_mlj == tlda_ms
    ## Check model traits
end

@testset "discriminant models checks" begin
    ## Data to be used for tests
    y = categorical(["apples", "oranges", "carrots", "mango"])
    X = (x1 =rand(4), x2 = collect(1:4))

    model = LDA()

    ## Note: The following tests depend on the order in which they are written.
    ## Hence do not change the ordering of the tests.

    ## Check to make sure error is thrown if we don't have more samples
    ## than unique classes during training.
    @test_throws ArgumentError fit(model, 1, X, y)

    ## Check to make sure error is thrown if `outdim` exceeds the number of features in
    ## sample matrix used in training.
    model = LDA(outdim=3)
    # categorical array with same pool as y but only containing "apples" & "oranges"
    y2 = y[[1,2,1,2]]
    @test_throws ArgumentError fit(model, 1, X, y2)

    ## Check to make sure error is thrown if UnivariateFinite `priors` doesn't have 
    ## common pool with target vector used in training.
    model = BayesianLDA(priors=UnivariateFinite([0.1, 0.5, 0.4], pool=missing))
    @test_throws ArgumentError fit(model, 1, X, y)

    ## Check to make sure error is thrown if keys used in `priors` dictionary are in pool 
    ## of training target used in training.
    model = BayesianLDA(priors=Dict("apples" => 0.1, "oranges"=>0.5, "bannana"=>0.4))
    @test_throws ArgumentError fit(model, 1, X, y)

    ## Check to make sure error is thrown if sum(`priors`) isn't approximately equal to 1.
    model = BayesianLDA(priors=UnivariateFinite(classes(y), [0.1, 0.5, 0.4, 0.2]))
    @test_throws ArgumentError fit(model, 1, X, y)

    ## Check to make sure error is thrown if `priors .< 0` or `priors .> 1`.
    model = BayesianLDA(priors=Dict(classes(y) .=> [-0.1, 0.0, 1.0, 0.1]))
    @test_throws ArgumentError fit(model, 1, X, y)
    model = BayesianLDA(priors=Dict(classes(y) .=> [1.1, 0.0, 0.0, -0.1]))
    @test_throws ArgumentError fit(model, 1, X, y)

    X2 = (x=rand(5),)
    y2 = coerce(collect("aaaaab"), Multiclass)[1:end - 1]
    corrected = MultivariateStats.SimpleCovariance(; corrected=true)

    ## Check that a lone target class in training is okay when using
    ## default hyperparameter values (issue #41):
    @test all([LDA, SubspaceLDA, BayesianLDA, BayesianSubspaceLDA]) do M
        model = M()
        Θ, _ = MLJBase.fit(model, 0, X2, y2)
        Set(levels(predict_mode(model, Θ, X2))) == Set(levels(y2))
    end

    ## Check that a lone target class in training is okay if `cov_w = SimpleCovariance(;
    ## corrected=true)` (issue #41):
    @test all([LDA, BayesianLDA]) do M
        model = M(cov_w=corrected)
        Θ, _ = MLJBase.fit(model, 0, X2, y2)
        Set(levels(predict_mode(model, Θ, X2))) == Set(levels(y2))
    end

    ## Check that an error is thrown in lone class case if `cov_b` is not
    ## `SimpleCovariance()` (see issue #41):
    for model in [
        LDA(cov_b=corrected),
        BayesianLDA(cov_b=corrected),
    ]
        @test_throws(
            MLJMultivariateStatsInterface.ERR_LONE_TARGET_CLASS,
            MLJBase.fit(model, 0, X2, y2)
        )
    end
end
