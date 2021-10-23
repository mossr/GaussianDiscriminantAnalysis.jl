module GaussianDiscriminantAnalysis

using Distributions
using Convex
using PyPlot
using SCS
using LinearAlgebra
using Random

export Input, Target, qda, lda, extract_data, get_data, get_positive_data, get_negative_data, plot_gda, generate_example_data

include("gda.jl")
include("lda.jl")
include("qda.jl")
include("svm.jl")
include("utils.jl")
include("plotting.jl")

end # module
