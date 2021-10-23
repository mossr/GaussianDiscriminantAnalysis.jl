# GaussianDiscriminantAnalysis.jl
[![Build Status](https://github.com/mossr/GaussianDiscriminantAnalysis.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mossr/GaussianDiscriminantAnalysis.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/mossr/GaussianDiscriminantAnalysis.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mossr/GaussianDiscriminantAnalysis.jl)

<p align="center">
    <img src="./img/cover.png">
</p>

## Installation
```julia
] add https://github.com/mossr/GaussianDiscriminantAnalysis.jl
```

## Quadratic discriminant analysis
```julia
using GaussianDiscriminantAnalysis

𝒟 = generate_example_data(100, seed=0) # ::Vector{Tuple{Array, Int}}
predict, mv_negative, mv_positive = qda(𝒟)
```

## Linear discriminant analysis
```julia
predict, mv_negative, mv_positive = lda(𝒟)
```

---

[Robert Moss](https://github.com/mossr)