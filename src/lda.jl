"""
Linear discriminant analysis.
`k` = which class shared their covariance.
"""
function lda(𝒟; kwargs...)
    neg_data = get_negative_data(𝒟)
    pos_data = get_positive_data(𝒟)
    return lda(neg_data, pos_data; kwargs...)
end


function lda(neg_data, pos_data; priors=[0.5, 0.5], soft=true, k=1)
    lda_func = soft ? lda_soft : lda_hard
    predict, mv_negative, mv_positive = gda(lda_func, neg_data, pos_data; priors=priors, k=k)
    return predict, mv_negative, mv_positive
end


function lda_soft(μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
    Σ = Σ₀ # shared covariances: doesn't matter which k we choose, covariance is copied/duplicated above.
    predict = x -> (x - μ₀)'inv(Σ)*(x - μ₀) - log(π₀) - (x - μ₁)'inv(Σ)*(x - μ₁) + log(π₁)
    return predict
end


function lda_hard(μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
    Σ = Σ₀ # shared covariances: doesn't matter which k we choose, covariance is copied/duplicated above.
    predictₖ = (x, μₖ, πₖ) -> x'inv(Σ)*μₖ - 1/2*μₖ'inv(Σ)*μₖ + log(πₖ)
    predict0 = x -> predictₖ(x, μ₀, π₀)
    predict1 = x -> predictₖ(x, μ₁, π₁)
    predict = x -> predict0(x) > predict1(x) ? -1 : 1
    return predict
end
