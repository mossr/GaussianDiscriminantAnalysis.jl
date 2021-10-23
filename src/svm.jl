# https://jump.dev/Convex.jl/v0.13.2/examples/general_examples/svm/
function compute_svm(pos_data, neg_data, solver=() -> SCS.Optimizer(verbose=0))
    # Create variables for the separating hyperplane w'*x = b.
    n = 2 # dimensionality of data
    C = 10 # inverse regularization parameter in the objective
    w = Variable(n)
    b = Variable()
    # Form the objective.
    obj = sumsquares(w) + C*sum(max(1+b-w'*pos_data, 0)) + C*sum(max(1-b+w'*neg_data, 0))
    # Form and solve problem.
    problem = minimize(obj)
    solve!(problem, solver)
    return evaluate(w), evaluate(b)
end


"""
Support vector machine
"""
function svm(pos_data, neg_data)
    x1_positive::Vector{Real} = pos_data[1,:]
    x1_negative::Vector{Real} = neg_data[1,:]
    svm_x = range(min(minimum(x1_positive), minimum(x1_negative)), stop=max(maximum(x1_positive), maximum(x1_negative)), length=1000)

    w, b = compute_svm(pos_data, neg_data)
    svm_boundary = (x,w,b) -> (-w[1] * x .+ b)/w[2] # line of the decision boundary
    svm_y = svm_boundary(svm_x, w, b)
    svm_classify = x -> sign((-w'*x + b)/w[2]) > 0 ? 1 : -1 # was : 0

    return svm_classify, svm_x, svm_y
end
