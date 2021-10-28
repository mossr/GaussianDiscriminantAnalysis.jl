const Input = Union{AbstractArray, Tuple}
const Target = Union{Bool, Int}


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


"""
Return all of the positive/negative data as a Matrix.
"""
get_data(𝒟, y) = hcat(first.(filter(data->data[2] == y, 𝒟))...)
get_negative_data(𝒟) = get_data(𝒟, 0)
get_positive_data(𝒟) = get_data(𝒟, 1)


function analyze_fit_svm(svm_classify, pos_data, neg_data)
    svm_true_positives = sum([svm_classify(x) for x in eachcol(pos_data)]) / size(pos_data)[2]
    svm_true_negatives = sum([1-svm_classify(x) for x in eachcol(neg_data)]) / size(neg_data)[2] # TODO: not quite right
    @info "SVM true positives: $(round(svm_true_positives, digits=4))"
    @info "SVM true negatives: $(round(svm_true_negatives, digits=4))"
end


function analyze_fit_gda(predict, pos_data, neg_data)
    gda_true_positives = sum([predict(x) > 0 ? 0 : 1 for x in eachcol(pos_data)]) / size(pos_data)[2]
    gda_true_negatives = sum([predict(x) <= 0 ? 0 : 1 for x in eachcol(neg_data)]) / size(neg_data)[2]
    @info "GDA true positives: $(round(gda_true_positives, digits=4))"
    @info "GDA true negatives: $(round(gda_true_negatives, digits=4))"
end


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