"""
Quadratic discriminant analysis.
"""
function qda(𝒟; kwargs...)
    neg_data = get_negative_data(𝒟)
    pos_data = get_positive_data(𝒟)
    return qda(neg_data, pos_data; kwargs...)
end


function qda(neg_data, pos_data; priors=[0.5, 0.5], soft=true)
    qda_func = soft ? qda_soft : qda_hard
    predict, mv_negative, mv_positive = gda(qda_func, neg_data, pos_data; priors=priors)
    return predict, mv_negative, mv_positive
end


function qda_soft(μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
    predict = x -> (x - μ₀)'inv(Σ₀)*(x - μ₀) + log(det(Σ₀)) - log(π₀) - (x - μ₁)'inv(Σ₁)*(x - μ₁) - log(det(Σ₁)) + log(π₁)
    return predict
end


function qda_hard(μ₀, Σ₀, π₀, μ₁, Σ₁, π₁)
    # QDA: in the form for class k=1
    predictₖ = (x, μₖ, Σₖ, πₖ) -> -1/2*log(det(Σₖ)) - 1/2*(x - μₖ)'inv(Σₖ)*(x - μₖ) + log(πₖ)
    predict0 = x -> predictₖ(x, μ₀, Σ₀, π₀)
    predict1 = x -> predictₖ(x, μ₁, Σ₁, π₁)
    predict = x -> predict0(x) > predict1(x) ? -1 : 1
    return predict
end
