X, y = @load_crabs

@testset "PCA" begin
    X_array = matrix(X)
    variance_ratio = 0.9999
    # MultivariateStats PCA
    pca_ms = MultivariateStats.fit(
        MultivariateStats.PCA,
        permutedims(X_array),
        pratio=variance_ratio
    )
    # MLJ PCA
    pca_mlj = PCA(variance_ratio=variance_ratio)
    _, _, report = test_decomposition_model(pca_ms, pca_mlj, X, X_array)
    
    # Test report
    @test report.indim == size(pca_ms)[1]
    @test report.outdim == size(pca_ms)[2]
    @test report.tprincipalvar == MS.tprincipalvar(pca_ms)
    @test report.tresidualvar == MS.tresidualvar(pca_ms)
    @test report.tvar == MS.var(pca_ms)
    @test report.mean == MS.mean(pca_ms)
    @test report.principalvars == MS.principalvars(pca_ms)
    @test report.loadings == MS.loadings(pca_ms)
end

@testset "KernelPCA" begin
    X_array = matrix(X)
    # MultivariateStats KernelPCA
    kpca_ms = MultivariateStats.fit(
        MultivariateStats.KernelPCA, permutedims(X_array),
        inverse=true
    )
    # MLJ KernelPCA
    kpca_mlj = KernelPCA()
    _, _, report = test_decomposition_model(kpca_ms, kpca_mlj, X, X_array)
    
    # Test report
    @test report.indim == size(kpca_ms)[1]
    @test report.outdim == size(kpca_ms)[2]
    @test report.principalvars == MS.eigvals(kpca_ms)
end

@testset "ICA" begin
    X_array = matrix(X)
    outdim = 5
    tolerance = 5.0
    # MultivariateStats ICA
    rng = StableRNG(1234) # winit gets randomly initialised
    #Random.seed!(1234) # winit gets randomly initialised
    ica_ms = MultivariateStats.fit(
        MultivariateStats.ICA,
        permutedims(X_array),
        outdim;
        tol=tolerance,
        winit = randn(rng, eltype(X_array), size(X_array, 2), outdim)
    )
    # MLJ ICA
    rng = StableRNG(1234) # winit gets randomly initialised
    #Random.seed!(1234) # winit gets randomly initialised
    ica_mlj = ICA(
        outdim=outdim,
        tol=tolerance,
        winit=randn(rng, eltype(X_array), size(X_array, 2), outdim))
    _, _, report = test_decomposition_model(
        ica_ms, ica_mlj, X, X_array, test_inverse=false
    )

    # Test report
    @test report.indim == size(ica_ms)[1]
    @test report.outdim == size(ica_ms)[2]
    @test report.mean == MS.mean(ica_ms)
end

@testset "ICA2" begin
    X_array = matrix(X)
    outdim = 5
    tolerance = 5.0
    # MultivariateStats ICA
    rng = StableRNG(1234) # winit gets randomly initialised
    #Random.seed!(1234) # winit gets randomly initialised
    ica_ms = MultivariateStats.fit(
        MultivariateStats.ICA,
        permutedims(X_array),
        outdim;
        tol=tolerance,
        fun=MultivariateStats.Gaus(),
        winit = randn(rng, eltype(X_array), size(X_array, 2), outdim)
    )
    # MLJ ICA
    rng = StableRNG(1234) # winit gets randomly initialised
    #Random.seed!(1234) # winit gets randomly initialised
    ica_mlj = ICA(
        outdim=outdim,
        tol=tolerance,
        fun=:gaus,
        winit=randn(rng, eltype(X_array), size(X_array, 2), outdim))
    test_decomposition_model(
        ica_ms, ica_mlj, X, X_array;
        test_inverse=false
    )
end

@testset "PPCA" begin
    X_array = matrix(X)
    tolerance = 5.0
    # MultivariateStats PPCA
    ppca_ms = MultivariateStats.fit(
        MultivariateStats.PPCA,
        permutedims(X_array);
        tol=tolerance
    )
    # MLJ PPCA
    ppca_mlj = PPCA(;tol=tolerance)
    _, _, report = test_decomposition_model(ppca_ms, ppca_mlj, X, X_array)
    
    # Test report
    @test report.indim == size(ppca_ms)[1]
    @test report.outdim == size(ppca_ms)[2]
    @test report.tvar == MS.var(ppca_ms)
    @test report.mean == MS.mean(ppca_ms)
    @test report.loadings == MS.loadings(ppca_ms)
end

@testset "FactorAnalysis" begin
    X_array = matrix(X)
    tolerance = 5.0
    eta = 0.01
    # MultivariateStats FactorAnalysis
    factoranalysis_ms = MultivariateStats.fit(
        MultivariateStats.FactorAnalysis,
        permutedims(X_array);
        tol=tolerance,
        η=eta
    )
    factoranalysis_mlj = FactorAnalysis(;tol=tolerance, eta=eta)
    _, _, report = test_decomposition_model(
        factoranalysis_ms, factoranalysis_mlj, X, X_array
    )

    # Test report
    @test report.indim == size(factoranalysis_ms)[1]
    @test report.outdim == size(factoranalysis_ms)[2]
    @test report.variance == MS.var(factoranalysis_ms)
    @test report.covariance_matrix == MS.cov(factoranalysis_ms)
    @test report.mean == MS.mean(factoranalysis_ms)
    @test report.loadings == MS.loadings(factoranalysis_ms)
end

