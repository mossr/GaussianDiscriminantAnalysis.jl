using Test
using GaussianDiscriminantAnalysis

@test begin
    𝒟 = generate_example_data(100, seed=0)
    x1_positive = extract_data(𝒟, 1; dims=1)
    x_positive = extract_data(𝒟, 1)
    x_negative = extract_data(𝒟, 0)
    predict, mv_negative, mv_positive = qda(𝒟)
    predict, mv_negative, mv_positive = lda(𝒟)
    plot_gda(𝒟)
    plot_gda(𝒟, soft=false)
    plot_gda(𝒟, use_qda=false)
    plot_gda(𝒟, use_qda=false, k=2)
    plot_gda(𝒟, use_qda=false, soft=false)
    plot_gda(𝒟, use_qda=false, soft=false, k=2)
    plot_gda(𝒟, subplots=true, show_svm=true, show_legend=true, show_analysis=true)
    plot_gda(𝒟, show_axes=false)
    plot_gda(𝒟, flip_colors=false)
    true
end
