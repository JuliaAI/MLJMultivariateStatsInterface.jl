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
    test_composition_model(pca_ms, pca_mlj, X, X_array)
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
    test_composition_model(kpca_ms, kpca_mlj, X, X_array)
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
    test_composition_model(ica_ms, ica_mlj, X, X_array, test_inverse=false)
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
    test_composition_model(ica_ms, ica_mlj, X, X_array, test_inverse=false)
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
    test_composition_model(ppca_ms, ppca_mlj, X, X_array)
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
    test_composition_model(factoranalysis_ms, factoranalysis_mlj, X, X_array)
end

