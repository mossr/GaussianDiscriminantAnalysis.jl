### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ c5be1b1c-468e-4994-aa71-e02f4ba5807c
using Distributions

# ╔═╡ 95983c33-b1e2-432e-81d6-65c36f6eeb06
using ColorSchemes

# ╔═╡ cd9e0b73-eba4-47d8-8348-a1c4354b85de
using Convex

# ╔═╡ 2871f476-a945-41fc-af0e-e00f51618d9a
using Plots

# ╔═╡ 191e0002-f1d8-45fe-95a9-311590646770
using SCS

# ╔═╡ 4198c1aa-0372-486b-bade-f22a55001f5f
using LinearAlgebra

# ╔═╡ a3a3d53f-3dca-44e0-adb7-7a0d6d48ad54
using Random

# ╔═╡ b0e0714c-56f0-426f-9de7-9c7234a73510
md"""
# GaussianDiscriminantAnalysis.jl
This file contains the entire GDA package code and include visualizations when viewing as a Pluto notebook.

Source: [https://github.com/mossr/GaussianDiscriminantAnalysis.jl](https://github.com/mossr/GaussianDiscriminantAnalysis.jl)
"""

# ╔═╡ 16176cd3-03e7-4c5f-a2e5-6a08e9b1b687
md"""
## GDA: Gaussian Discriminant Analysis
"""

# ╔═╡ 4415d9a9-7777-44c6-ae13-ae79d7ef1d99
"""
	mv_fit(neg_data, pos_data, k)::Tuple

Fits multivariate Gaussian distributions to both datasets. Optionally specify which class `k` shares its covariance.
"""
function mv_fit(neg_data, pos_data, k=missing)
    mv_negative = fit_mle(MvNormal, neg_data)
    mv_positive = fit_mle(MvNormal, pos_data)

    if !ismissing(k)
        # LDA with shared covariances
        if k == 1 # which class k shared their covariance?
            mv_positive = MvNormal(mv_positive.μ, mv_negative.Σ) # shared covariance
        else
            mv_negative = MvNormal(mv_negative.μ, mv_positive.Σ) # shared covariance
        end
    end

    return (mv_negative, mv_positive)
end

# ╔═╡ 78ff7d8f-93e1-4630-bb91-7ca5187434e1
"""
Return mean `μ` and covariance `Σ` from a single `MvNormal`.
"""
extract_parameters(mv::MvNormal) = (mv.μ, mv.Σ)

# ╔═╡ d29cdec3-e18e-4566-842b-2f8b5ff7eae9
"""
Return mean, covariance, and priors for both input `MvNormal` distributions.
"""
function extract_parameters(mv_negative, mv_positive, priors)
	# Class 0 = negative, Class 1 = positive
	@assert sum(priors) ≈ 1
	π₀, π₁ = priors
	μ₀, Σ₀ = extract_parameters(mv_negative)
	μ₁, Σ₁ = extract_parameters(mv_positive)
	return (μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
end

# ╔═╡ 5e32df5f-c104-4d7b-b778-f1b648294195
"""
**Gaussian discriminant analysis.** Returns a function that predicts the class of an input \$x\$. Provide one of the following classification functions (`gda_func`):
- `qda_soft`
- `qda_hard`
- `lda_soft`
- `lda_hard`

See the `qda` and `lda` functions.
"""
function gda(gda_func, neg_data, pos_data; priors=[0.5, 0.5], k=missing)
    mv_negative, mv_positive = mv_fit(neg_data, pos_data, k)
    (μ₀, Σ₀, π₀, μ₁, Σ₁, π₁) = extract_parameters(mv_negative, mv_positive, priors)
    predict = gda_func(μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
    return predict, mv_negative, mv_positive
end

# ╔═╡ c560b881-eb53-4bba-bef1-5e90b90579b7
md"""
## LDA: Linear Discriminant Analysis
"""

# ╔═╡ cc74dd31-0ffb-4af1-9185-6a2c7ac0d4e4
"""
_Soft linear discriminant analysis_, provides prediction function `δ` that gives a signed real-valued score to which class the input `x` is closest to/classified as.

\$\\begin{align}\\delta(x) = &(x - \\mu_0)^\\top\\Sigma^{-1}(x - \\mu_0) - \\log\\pi_0 - \\\\&(x - \\mu_1)^\\top\\Sigma^{-1}(x - \\mu_1) + \\log \\pi_1\\end{align}\$
"""
function lda_soft(μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
	# Shared covariances: doesn't matter which k, covariance is copied above.
    Σ = Σ₀
    δ = x -> (x - μ₀)'inv(Σ)*(x - μ₀) - log(π₀) - (x - μ₁)'inv(Σ)*(x - μ₁) + log(π₁)
    return δ
end

# ╔═╡ 058527ba-b6ec-453f-90af-b2e5c27199f9
"""
_Hard linear discriminant analysis_, provides prediction function `δ` that gives either -1 or 1 to which class the input `x` is closest to/classified as.

\$\\begin{equation}
\\delta_k(x) = x^\\top\\Sigma^{-1}\\mu_k - \\frac{1}{2} \\mu_k^\\top\\Sigma^{-1}\\mu_k + \\log\\pi_k
\\end{equation}\$
"""
function lda_hard(μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
    Σ = Σ₀ # shared covariances: doesn't matter which k we choose, covariance is copied/duplicated above.
    predictₖ = (x, μₖ, πₖ) -> x'inv(Σ)*μₖ - 1/2*μₖ'inv(Σ)*μₖ + log(πₖ)
    predict0 = x -> predictₖ(x, μ₀, π₀)
    predict1 = x -> predictₖ(x, μ₁, π₁)
    predict = x -> predict0(x) > predict1(x) ? -1 : 1
    return predict
end

# ╔═╡ 5f4f55be-6c72-4a28-9179-760c2657938c
md"""
## QDA: Quadratic Discriminant Analysis
"""

# ╔═╡ b05789d7-21a1-4fbd-a612-9f1e9d23c788
"""
_Soft quadratic discriminant analysis_, provides prediction function `δ` that gives a signed real-valued score to which class the input `x` is closest to/classified as.

\$\\begin{align}\\delta(x) = &(x - \\mu_0)^\\top\\Sigma_0^{-1}(x - \\mu_0) + \\log|\\Sigma_0| - \\log\\pi_0 - \\\\&(x - \\mu_1)^\\top\\Sigma_1^{-1}(x - \\mu_1) - \\log|\\Sigma_1| + \\log \\pi_1\\end{align}\$
"""
function qda_soft(μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
    δ = x -> (x - μ₀)'inv(Σ₀)*(x - μ₀) + log(det(Σ₀)) - log(π₀) -
	         (x - μ₁)'inv(Σ₁)*(x - μ₁) - log(det(Σ₁)) + log(π₁)
    return δ
end

# ╔═╡ f75ba523-6186-4d75-877a-691e473f2e7e
"""
_Hard quadratic discriminant analysis_, provides prediction function `δ` that gives either -1 or 1 to which class the input `x` is closest to/classified as.

\$\\begin{equation}
\\delta_k(x) = -\\frac{1}{2} \\log(|\\Sigma_k|) - \\frac{1}{2} (x - \\mu_k)^\\top\\Sigma_k^{-1}(x - \\mu_k) + \\log\\pi_k
\\end{equation}\$
"""
function qda_hard(μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
    # QDA: in the form for class k=1
    predictₖ = (x,μₖ,Σₖ,πₖ) -> -1/2*log(det(Σₖ)) - 1/2*(x-μₖ)'inv(Σₖ)*(x-μₖ) + log(πₖ)
    predict0 = x -> predictₖ(x, μ₀, Σ₀, π₀)
    predict1 = x -> predictₖ(x, μ₁, Σ₁, π₁)
    predict = x -> predict0(x) > predict1(x) ? -1 : 1
    return predict
end

# ╔═╡ 2701f016-aa47-42aa-b619-f2613c2972f8
md"""
## SVM: Support Vector Machine
"""

# ╔═╡ 42d3e375-7eb1-4028-8b61-d5f5b261f075
"""
Compute the SVM weights and biases of the input data. Taken from [Convevx.jl example](https://jump.dev/Convex.jl/v0.13.2/examples/general_examples/svm/).

\$\\operatorname*{minimize}_{\\mathbf{w}, b} \\quad
\\begin{align}
\\lVert \\mathbf w \\lVert^2 + C * \\left( \\sum_{i=1}^N \\max(1 + b - \\mathbf w^\\top \\mathbf x_i, 0) + \\sum_{i=1}^M \\max(1 - b + \\mathbf w^\\top \\mathbf y_i, 0) \\right)
\\end{align}\$
"""
function compute_svm(pos_data, neg_data, solver=() -> SCS.Optimizer(verbose=0))
    # Create variables for the separating hyperplane w'*x = b.
    n = 2 # dimensionality of data
    C = 10 # inverse regularization parameter in the objective
    w = Variable(n)
    b = Variable()
    # Form the objective.
    obj = sumsquares(w) + C * (sum(max(1+b-w'*pos_data,0)) +
	                           sum(max(1-b+w'*neg_data,0)))
    # Form and solve problem.
    problem = minimize(obj)
    solve!(problem, solver)
    return evaluate(w), evaluate(b)
end

# ╔═╡ 7abaf0d0-7162-46ab-a009-b44f16162a6c
"""
_Support vector machine._ Given the positive and negative datapoints, return the classification function and boundary line that each take an input `x`.

	svm(pos_data, neg_data)::Tuple{Function, Function}

\$\\delta(\\mathbf x) = \\frac{-w_1 \\mathbf{x} + b}{w_2}\$
\$\\operatorname{classify}(\\mathbf{x}) = \\operatorname{sign}\\left(\\frac{-\\mathbf{w}^\\top \\mathbf{x} + b}{w_2}\\right)\$
"""
function svm(pos_data, neg_data)
    x1_positive::Vector{Real} = pos_data[1,:]
    x1_negative::Vector{Real} = neg_data[1,:]
    svm_x = range(min(minimum(x1_positive), minimum(x1_negative)),
		          stop=max(maximum(x1_positive), maximum(x1_negative)), length=1000)
    w, b = compute_svm(pos_data, neg_data)
    svm_boundary = x -> (-w[1]*x .+ b)/w[2] # decision boundary line (closures w, b)
    svm_classify = x -> sign((-w'x + b)/w[2]) > 0 ? 1 : -1 # was : 0

    return svm_classify, svm_boundary
end

# ╔═╡ 4033b62a-19ed-465c-a3fe-d4f4a4d3f005
md"""
## Utilities
"""

# ╔═╡ b7994c52-5363-482c-9024-4b6760633830
const Input = Union{AbstractArray, Tuple}

# ╔═╡ 0ab77f93-afce-4b8d-8641-525d18285deb
const Target = Union{Bool, Int}

# ╔═╡ 539decb8-5bc4-4af0-9887-71c2c421239d
"""
Extract components of the input 𝐱 given a target y ∈ {0, 1} value.
`dims` specifies the dimension of 𝐱 to extract, defaulting to the entire 𝐱.
"""
function extract_data(𝒟, target; dims=missing)
    extracted = []
    for data in 𝒟
        𝐱, y = data
        if target == y
            if ismissing(dims)
                push!(extracted, 𝐱)
            else
                push!(extracted, 𝐱[dims])
            end
        end
    end
    return extracted
end

# ╔═╡ 97c22a89-5a09-47d8-80e8-f175739ac4e3
"""
Return all of the positive/negative data as a Matrix.
"""
get_data(𝒟, y) = hcat(first.(filter(data->data[2] == y, 𝒟))...)

# ╔═╡ 04b5a4ad-bb5a-4b22-9fcd-9002dc3e94f5
"""
Strip out the negative data from `𝒟`.
"""
get_negative_data(𝒟) = get_data(𝒟, 0)

# ╔═╡ 6d7ca4b8-5c2f-425f-b578-1d2ac8d3db75
"""
Strip out the positive data from `𝒟`.
"""
get_positive_data(𝒟) = get_data(𝒟, 1)

# ╔═╡ beea0afb-da9b-48b4-bc8e-429837003a43
begin
	"""
	**Linear discriminant analysis.**
	Uses a single covariance to predict a linear boundary between the classes.

		lda(𝒟)::Tuple{Function, MvNormal, MvNormal}
	"""
	function lda(𝒟; kwargs...)
	    neg_data = get_negative_data(𝒟)
	    pos_data = get_positive_data(𝒟)
	    return lda(neg_data, pos_data; kwargs...)
	end

	"""
	**Linear discriminant analysis.**

		lda(neg_data, pos_data; priors, soft, k)::Tuple

	- `priors` = defaults to uniform.
	- `soft` = determins ''soft'' or ''hard'' LDA (default `true`)
	- `k` = which class shared their covariance.

	"""
	function lda(neg_data, pos_data; priors=[0.5, 0.5], soft=true, k=1)
	    lda_func = soft ? lda_soft : lda_hard
	    predict, mv_negative, mv_positive = gda(lda_func, neg_data, pos_data;
												priors=priors, k=k)
	    return predict, mv_negative, mv_positive
	end
end

# ╔═╡ 221f711c-8bd7-47c4-85fb-65b942c31099
begin
	"""
	**Quadratic discriminant analysis.**
	Uses both covariances to predict a quadratic boundary between the classes.

		qda(𝒟)::Tuple{Function, MvNormal, MvNormal}
	"""
	function qda(𝒟; kwargs...)
	    neg_data = get_negative_data(𝒟)
	    pos_data = get_positive_data(𝒟)
	    return qda(neg_data, pos_data; kwargs...)
	end

	"""
	**Quadratic discriminant analysis.**

		qda(neg_data, pos_data; priors, soft, k)::Tuple

	- `priors` = defaults to uniform.
	- `soft` = determins ''soft'' or ''hard'' QDA (default `true`)
	- `k` = which class shared their covariance.

	"""
	function qda(neg_data, pos_data; priors=[0.5, 0.5], soft=true)
	    qda_func = soft ? qda_soft : qda_hard
	    predict, mv_negative, mv_positive = gda(qda_func, neg_data, pos_data;
			                                    priors=priors)
	    return predict, mv_negative, mv_positive
	end
end

# ╔═╡ cd244d90-54ad-4e47-93e8-2730553eab5c
"""
Print out fit metrics for SVM true positives and true negatives.
"""
function analyze_fit_svm(svm_classify, pos_data, neg_data)
    svm_true_positives = 
		sum([svm_classify(x) for x in eachcol(pos_data)]) / size(pos_data)[2]
    svm_true_negatives =
		sum([1-svm_classify(x) for x in eachcol(neg_data)]) / size(neg_data)[2] # TODO: not quite right
    @info "SVM true positives: $(round(svm_true_positives, digits=4))"
    @info "SVM true negatives: $(round(svm_true_negatives, digits=4))"
end

# ╔═╡ cd816c42-8ffe-4c85-8ab4-1f190c5648fe
"""
Print out fit metrics for GDA true positives and true negatives.
"""
function analyze_fit_gda(predict, pos_data, neg_data)
    gda_true_positives =
		sum([predict(x) > 0 ? 0 : 1 for x in eachcol(pos_data)]) / size(pos_data)[2]
    gda_true_negatives =
		sum([predict(x) <= 0 ? 0 : 1 for x in eachcol(neg_data)]) / size(neg_data)[2]
    @info "GDA true positives: $(round(gda_true_positives, digits=4))"
    @info "GDA true negatives: $(round(gda_true_negatives, digits=4))"
end

# ╔═╡ 3e78fdde-007c-426f-a83e-d11610ba55a3
"""
Generate example labeled classification data that has a pretty multivariate fit.
"""
function generate_example_data(n=100; seed=missing)
    if !ismissing(seed)
        Random.seed!(seed)
    end

    mv_y0 = MvNormal([0.18, 6.0], [0.04 0; 0 0.7] .^ 2)
    mv_y1 = MvNormal([0.2, 3.5], [0.15 0; 0 0.1] .^ 2)
    y0 = rand(mv_y0, n)
    y1 = rand(mv_y1, n)

    𝒟 = vcat([(Vector(𝐱), 1) for 𝐱 in eachcol(y1)],
             [(Vector(𝐱), 0) for 𝐱 in eachcol(y0)])

    return 𝒟
end

# ╔═╡ db1fb757-af9c-48b9-84d5-b33dd113b30e
md"""
## Plotting
"""

# ╔═╡ 4ccfcbce-214e-4396-a542-ca04e3df43b9
global NEGATIVE_COLOR = "#d62728"

# ╔═╡ 61bba33e-e3b3-4469-a010-68447e4ce939
global POSITIVE_COLOR = "#2ca02c"

# ╔═╡ 1467dd57-0191-4fd6-9bdc-0c21d6c0acec
"""
```
gdaplot(𝒟;
        soft=true,            # soft/hard prediction boundary
        use_qda=true,         # QDA or LDA
        k=1,                  # specify the class k to share covariance (LDA)
        rev=false,            # reverse "positive" and "negative"
        heatmap=false,        # use heatmap instead of filled contours
        levels=100,           # number of levels for the filled contours
        show_axes=true,       # toggle displaying of axes
        subplots=false,       # include 1D Gaussian fits in subplots
        show_svm=false,       # show SVM decision boundary
        show_analysis=false,  # print out goodness of prediction
        show_legend=true,     # toggle showing of legend
        return_predict=false) # return (fig, predict) instead of just (fig)
```
"""
function gdaplot(𝒟;
        soft=true,            # soft/hard prediction boundary
        use_qda=true,         # QDA or LDA
        k=1,                  # specify the class k to share covariance (LDA)
        rev=false,            # reverse "positive" and "negative"
        heatmap=false,        # use heatmap instead of filled contours
        levels=100,           # number of levels for the filled contours
        show_axes=true,       # toggle displaying of axes
        subplots=false,       # include 1D Gaussian fits in subplots
        show_svm=false,       # show SVM decision boundary
        show_analysis=false,  # print out goodness of prediction
        show_legend=true,     # toggle showing of legend
        return_predict=false) # return (fig, predict) instead of just (fig)

    default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

    pos_data = get_positive_data(𝒟)
    neg_data = get_negative_data(𝒟)

    x1_positive::Vector{Real} = pos_data[1,:]
    x2_positive::Vector{Real} = pos_data[2,:]
    x1_negative::Vector{Real} = neg_data[1,:]
    x2_negative::Vector{Real} = neg_data[2,:]

    # select the type of GDA we want to perform (LDA or QDA)
    if use_qda
        gda_func = qda
    else
        gda_func = (nd, pd; soft=soft) -> lda(nd, pd, soft=soft, k=k) # pass in k
    end

    #============================================#
    # run Gaussian discriminant analysis
    #============================================#
    predict, mv_negative, mv_positive = gda_func(neg_data, pos_data; soft=soft)

    #============================================#
    # plot prediction boundary contours
    #============================================#
    min_x1_pos = minimum(x1_positive)
    min_x1_neg = minimum(x1_negative)
    min_x1 = min(min_x1_pos, min_x1_neg)

    max_x1_pos = maximum(x1_positive)
    max_x1_neg = maximum(x1_negative)
    max_x1 = max(max_x1_pos, max_x1_neg)

    min_x2_pos = minimum(x2_positive)
    min_x2_neg = minimum(x2_negative)
    min_x2 = min(min_x2_pos, min_x2_neg)

    max_x2_pos = maximum(x2_positive)
    max_x2_neg = maximum(x2_negative)
    max_x2 = max(max_x2_pos, max_x2_neg)

    # adjust min/max to add margin
    scale = 0.1
    adjust_x1 = (max_x1 - min_x1) * scale
    adjust_x2 = (max_x2 - min_x2) * scale
    min_x1 -= adjust_x1
    max_x1 += adjust_x1
    min_x2 -= adjust_x2
    max_x2 += adjust_x2

    dbX = range(min_x1, stop=max_x1, length=1000)
    dbY = range(min_x2, stop=max_x2, length=1000)
    dbZ = [predict([x,y]) for y in dbY, x in dbX] # Note x-y "for" ordering
    vmin = minimum(dbZ)
    vmax = maximum(dbZ)

    buckets = [vmin, vmin/2, 0, vmax/2, vmax] # shift colormap so 0 is at center
    normed = (buckets .- vmin) / (vmax - vmin)
    cmap = cgrad(:RdYlGn, normed, rev=rev)

    if heatmap
        fig = Plots.heatmap(dbX, dbY, dbZ, c=cmap,
			                colorbar_entry=!(subplots || !show_legend))
    else
        fig = contourf(dbX, dbY, dbZ, c=cmap, levels=levels, linewidth=0,
			           colorbar_entry=!(subplots || !show_legend))
    end

    if rev
        positive_color_contour = :plasma
        negative_color_contour = :viridis
        positive_color = NEGATIVE_COLOR
        negative_color = POSITIVE_COLOR
    else
        positive_color_contour = :viridis
        negative_color_contour = :plasma
        positive_color = POSITIVE_COLOR
        negative_color = NEGATIVE_COLOR
    end

    current_ylim = ylims()
    current_xlim = xlims()

    scatter!(x1_negative, x2_negative, label="negative", alpha=0.5,
		     color=negative_color, ms=3, msc=:black, msw=2)
    scatter!(x1_positive, x2_positive, label="positive", alpha=0.5,
		     color=positive_color, ms=3, msc=:black, msw=2)

    #============================================#
    # plot multivariate Gaussian contours (positive)
    #============================================#
    pX = range(min_x1, stop=max_x1, length=1000)
    pY = range(min_x2, stop=max_x2, length=1000)
    pZ = [pdf(mv_positive, [x,y]) for y in pY, x in pX] # Note x-y "for" ordering
    contour!(pX, pY, pZ, lw=2, alpha=0.5,
		     color=positive_color_contour, colorbar_entry=false)

    #============================================#
    # plot multivariate Gaussian contours (negative)
    #============================================#
    nX = range(min_x1, stop=max_x1, length=1000)
    nY = range(min_x2, stop=max_x2, length=1000)
    nZ = [pdf(mv_negative, [x,y]) for y in nY, x in nX] # Note x-y "for" ordering
    contour!(nX, nY, nZ, lw=2, alpha=0.5,
		     color=negative_color_contour, colorbar_entry=false)

    # restore axis limits (i.e. tight layout of contourf/heatmap)
    xlims!(current_xlim)
    ylims!(current_ylim)

    if show_svm
        @info "Running SVM..."
        svm_classify, svm_boundary = svm(pos_data, neg_data)

        svm_y = svm_boundary(dbX)
        plot!(dbX, svm_y, label="SVM", color="black", lw=2)

        if show_analysis
            analyze_fit_svm(svm_classify, pos_data, neg_data)
        end
    end

    if subplots
        lay = @layout [a{0.3h} _; b{0.7h, 0.7w} c]

        topfig = histogram(x1_positive, color=positive_color, normalize=true,
			               alpha=0.5, label=nothing, xaxis=nothing)
        histogram!(x1_negative, color=negative_color, normalize=true,
			       alpha=0.5, label=nothing)
        normal_x1_positive = fit_mle(Normal, x1_positive) # NOTE: Try Gamma?
        normal_x1_negative = fit_mle(Normal, x1_negative)
        plot!(x->pdf(normal_x1_positive, x), color=positive_color,
			  xlim=xlims(), label=false, lw=2)
        plot!(x->pdf(normal_x1_negative, x), color=negative_color,
			  xlim=xlims(), label=false, lw=2)
        xlims!(current_xlim) # match limits of main plot

        sidefig = histogram(x2_positive, color=positive_color, normalize=true,
			                alpha=0.5, label=nothing, yaxis=nothing, orientation=:h)
        histogram!(x2_negative, color=negative_color, normalize=true,
			       alpha=0.5, label=nothing, orientation=:h)
        normal_x2_positive = fit_mle(Normal, x2_positive) # NOTE: Try Gamma?
        normal_x2_negative = fit_mle(Normal, x2_negative)
        plot!([pdf(normal_x2_positive, x) for x in pY], pY,
			  color=positive_color, xlim=xlims(), label=false, lw=2)
        plot!([pdf(normal_x2_negative, x) for x in nY], nY,
			  color=negative_color, xlim=xlims(), label=false, lw=2)
        ylims!(current_ylim) # match limits of main plot

        fig = plot(topfig, fig, sidefig, layout=lay)
    end

    if !show_axes
        plot!(axis=nothing)
    end

    if show_analysis
        analyze_fit_gda(predict, pos_data, neg_data)
    end

    if !show_legend
        plot!(legend=false)
    end

    return return_predict ? (fig, predict) : fig
end

# ╔═╡ 289086b7-c068-4f61-9639-20c52f41d5d1
md"""
## Notebook
"""

# ╔═╡ 05b4d1f7-8917-4510-a3d2-421a0c66f1cc
is_notebook = false

# ╔═╡ e69e96ec-f6d3-48ff-9c6f-d915f9c9a995
if is_notebook
	using PlutoUI
	𝒟 = generate_example_data(100, seed=0xC0FFEE)
end

# ╔═╡ f204d9da-3cab-4ca3-a597-50009e5cdf28
is_notebook && TableOfContents()

# ╔═╡ 07b0385b-b065-4e16-ba93-ee34457044c5
is_notebook && gdaplot(𝒟)

# ╔═╡ 1f09ea5c-5c57-435f-885f-2a4882782d51
is_notebook && gdaplot(𝒟, use_qda=false)

# ╔═╡ 219be453-b6e6-4150-a8f9-41a4f3d42951
is_notebook && gdaplot(𝒟, use_qda=false, soft=false)

# ╔═╡ bf7a2a72-8a17-4301-af80-9a2f0d8dff31
is_notebook && gdaplot(𝒟)

# ╔═╡ a74a168a-0147-4dea-9a9e-c0bb2762b73f
is_notebook && gdaplot(𝒟, soft=false)

# ╔═╡ 5307cc37-61dd-4eb7-a3b2-29ca009d37b5
is_notebook && gdaplot(𝒟, show_svm=true)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Convex = "f65535da-76fb-5f13-bab9-19810c17039a"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SCS = "c946c3f1-0d1f-5ce8-9dea-7daa1f7e2d13"

[compat]
ColorSchemes = "~3.15.0"
Convex = "~0.14.18"
Distributions = "~0.25.34"
Plots = "~1.24.3"
PlutoUI = "~0.7.21"
SCS = "~0.8.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AMD]]
deps = ["Libdl", "LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "fc66ffc5cff568936649445f58a55b81eaf9592c"
uuid = "14f7f29c-3bd6-536c-9a0b-7339e30b5a3e"
version = "0.4.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "abb72771fd8895a7ebd83d5632dc4b989b022b5b"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "61adeb0823084487000600ef8b1c00cc2474cd47"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.0"

[[deps.BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Convex]]
deps = ["AbstractTrees", "BenchmarkTools", "LDLFactorizations", "LinearAlgebra", "MathOptInterface", "OrderedCollections", "SparseArrays", "Test"]
git-tree-sha1 = "145c5e0b3ea3c9dd3bba134a58bab4112aa250c8"
uuid = "f65535da-76fb-5f13-bab9-19810c17039a"
version = "0.14.18"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "7f3bec11f4bcd01bc1f507ebce5eadf1b0a78f47"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.34"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LDLFactorizations]]
deps = ["AMD", "LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "399bbe845e06e1c2d44ebb241f554d45eaf66788"
uuid = "40e66cde-538c-5869-a4ad-c39174c6795b"
version = "0.8.1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "92b7de61ecb616562fd2501334f729cc9db2a9a6"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "0.10.6"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "7bb6853d9afec54019c1397c6eb610b9b9a19525"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.3.1"

[[deps.NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "d73736030a094e8d24fdf3629ae980217bf1d59d"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.24.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "b68904528fd538f1cb6a3fbc44d2abdc498f9e8e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.21"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SCS]]
deps = ["BinaryProvider", "Libdl", "LinearAlgebra", "MathOptInterface", "Requires", "SCS_GPU_jll", "SCS_jll", "SparseArrays"]
git-tree-sha1 = "c819d023621358f3c08f08d41bd9354cf1357d35"
uuid = "c946c3f1-0d1f-5ce8-9dea-7daa1f7e2d13"
version = "0.8.1"

[[deps.SCS_GPU_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "a96402e3b494a8bbec61b1adb86d4be04112c646"
uuid = "af6e375f-46ec-5fa0-b791-491b0dfa44a4"
version = "2.1.4+0"

[[deps.SCS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "6cdaccb5e6a69455f960de1ae445ba1de5db9d0d"
uuid = "f4f2fc5b-1d94-523c-97ea-2ab488bedf4b"
version = "2.1.2+1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─b0e0714c-56f0-426f-9de7-9c7234a73510
# ╠═c5be1b1c-468e-4994-aa71-e02f4ba5807c
# ╠═95983c33-b1e2-432e-81d6-65c36f6eeb06
# ╠═cd9e0b73-eba4-47d8-8348-a1c4354b85de
# ╠═2871f476-a945-41fc-af0e-e00f51618d9a
# ╠═191e0002-f1d8-45fe-95a9-311590646770
# ╠═4198c1aa-0372-486b-bade-f22a55001f5f
# ╠═a3a3d53f-3dca-44e0-adb7-7a0d6d48ad54
# ╠═f204d9da-3cab-4ca3-a597-50009e5cdf28
# ╠═07b0385b-b065-4e16-ba93-ee34457044c5
# ╟─16176cd3-03e7-4c5f-a2e5-6a08e9b1b687
# ╠═5e32df5f-c104-4d7b-b778-f1b648294195
# ╠═4415d9a9-7777-44c6-ae13-ae79d7ef1d99
# ╠═78ff7d8f-93e1-4630-bb91-7ca5187434e1
# ╠═d29cdec3-e18e-4566-842b-2f8b5ff7eae9
# ╟─c560b881-eb53-4bba-bef1-5e90b90579b7
# ╠═beea0afb-da9b-48b4-bc8e-429837003a43
# ╠═1f09ea5c-5c57-435f-885f-2a4882782d51
# ╠═cc74dd31-0ffb-4af1-9185-6a2c7ac0d4e4
# ╠═058527ba-b6ec-453f-90af-b2e5c27199f9
# ╠═219be453-b6e6-4150-a8f9-41a4f3d42951
# ╟─5f4f55be-6c72-4a28-9179-760c2657938c
# ╠═221f711c-8bd7-47c4-85fb-65b942c31099
# ╠═bf7a2a72-8a17-4301-af80-9a2f0d8dff31
# ╠═b05789d7-21a1-4fbd-a612-9f1e9d23c788
# ╠═f75ba523-6186-4d75-877a-691e473f2e7e
# ╠═a74a168a-0147-4dea-9a9e-c0bb2762b73f
# ╟─2701f016-aa47-42aa-b619-f2613c2972f8
# ╠═42d3e375-7eb1-4028-8b61-d5f5b261f075
# ╠═7abaf0d0-7162-46ab-a009-b44f16162a6c
# ╠═5307cc37-61dd-4eb7-a3b2-29ca009d37b5
# ╟─4033b62a-19ed-465c-a3fe-d4f4a4d3f005
# ╠═b7994c52-5363-482c-9024-4b6760633830
# ╠═0ab77f93-afce-4b8d-8641-525d18285deb
# ╠═539decb8-5bc4-4af0-9887-71c2c421239d
# ╠═97c22a89-5a09-47d8-80e8-f175739ac4e3
# ╠═04b5a4ad-bb5a-4b22-9fcd-9002dc3e94f5
# ╠═6d7ca4b8-5c2f-425f-b578-1d2ac8d3db75
# ╠═cd244d90-54ad-4e47-93e8-2730553eab5c
# ╠═cd816c42-8ffe-4c85-8ab4-1f190c5648fe
# ╠═3e78fdde-007c-426f-a83e-d11610ba55a3
# ╟─db1fb757-af9c-48b9-84d5-b33dd113b30e
# ╠═4ccfcbce-214e-4396-a542-ca04e3df43b9
# ╠═61bba33e-e3b3-4469-a010-68447e4ce939
# ╠═1467dd57-0191-4fd6-9bdc-0c21d6c0acec
# ╟─289086b7-c068-4f61-9639-20c52f41d5d1
# ╠═05b4d1f7-8917-4510-a3d2-421a0c66f1cc
# ╠═e69e96ec-f6d3-48ff-9c6f-d915f9c9a995
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
