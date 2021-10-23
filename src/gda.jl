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


extract_parameters(mv::MvNormal) = (mv.μ, mv.Σ)

function extract_parameters(mv_negative, mv_positive, priors)
    # Class 0 = negative, Class 1 = positive
    @assert sum(priors) ≈ 1
    π₀, π₁ = priors
    μ₀, Σ₀ = extract_parameters(mv_negative)
    μ₁, Σ₁ = extract_parameters(mv_positive)
    return (μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
end


"""
Gaussian discriminant analysis: provide a `gda_func` of either
`qda_soft`, `qda_hard`, `lda_soft`, or `lda_hard`.

See the `qda` and `lda` functions.
"""
function gda(gda_func, neg_data, pos_data; priors=[0.5, 0.5], k=missing)
    mv_negative, mv_positive = mv_fit(neg_data, pos_data, k)
    (μ₀, Σ₀, π₀, μ₁, Σ₁, π₁) = extract_parameters(mv_negative, mv_positive, priors)
    predict = gda_func(μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
    return predict, mv_negative, mv_positive
end
