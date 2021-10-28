global NEGATIVE_COLOR = "#d62728"
global POSITIVE_COLOR = "#2ca02c"


function gdaplot(𝒟;
        soft=true,            # soft/hard prediction boundary
        use_qda=true,         # QDA or LDA
        k=1,                  # specify which class k to share covariance (LDA only)
        rev=false,            # reverse "positive" and "negative"
        heatmap=false,        # use heatmap instead of filled contours
        levels=100,           # number of levels for the filled contours
        show_axes=true,       # toggle displaying of axes
        subplots=false,       # include single-dimensional Gaussian fits in subplots
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
        fig = Plots.heatmap(dbX, dbY, dbZ, c=cmap, colorbar_entry=!(subplots || !show_legend))
    else
        fig = contourf(dbX, dbY, dbZ, c=cmap, levels=levels, linewidth=0, colorbar_entry=!(subplots || !show_legend))
    end

    if rev
        positive_color_contour = :plasma
        negative_color_contour = :viridis
        positive_color = GaussianDiscriminantAnalysis.NEGATIVE_COLOR
        negative_color = GaussianDiscriminantAnalysis.POSITIVE_COLOR
    else
        positive_color_contour = :viridis
        negative_color_contour = :plasma
        positive_color = GaussianDiscriminantAnalysis.POSITIVE_COLOR
        negative_color = GaussianDiscriminantAnalysis.NEGATIVE_COLOR
    end

    current_ylim = ylims()
    current_xlim = xlims()

    scatter!(x1_negative, x2_negative, label="negative", alpha=0.5, color=negative_color, ms=3, msc=:black, msw=2)
    scatter!(x1_positive, x2_positive, label="positive", alpha=0.5, color=positive_color, ms=3, msc=:black, msw=2)

    #============================================#
    # plot multivariate Gaussian contours (positive)
    #============================================#
    pX = range(min_x1, stop=max_x1, length=1000)
    pY = range(min_x2, stop=max_x2, length=1000)
    pZ = [pdf(mv_positive, [x,y]) for y in pY, x in pX] # Note x-y "for" ordering
    contour!(pX, pY, pZ, lw=2, alpha=0.5, color=positive_color_contour, colorbar_entry=false)

    #============================================#
    # plot multivariate Gaussian contours (negative)
    #============================================#
    nX = range(min_x1, stop=max_x1, length=1000)
    nY = range(min_x2, stop=max_x2, length=1000)
    nZ = [pdf(mv_negative, [x,y]) for y in nY, x in nX] # Note x-y "for" ordering
    contour!(nX, nY, nZ, lw=2, alpha=0.5, color=negative_color_contour, colorbar_entry=false)

    # restore axis limits (i.e. tight layout of contourf/heatmap)
    xlims!(current_xlim)
    ylims!(current_ylim)

    if show_svm
        @info "Running SVM..."
        svm_classify, svm_boundary = GaussianDiscriminantAnalysis.svm(pos_data, neg_data)

        svm_y = svm_boundary(dbX)
        plot!(dbX, svm_y, label="SVM", color="black", lw=2)

        if show_analysis
            analyze_fit_svm(svm_classify, pos_data, neg_data)
        end
    end

    if subplots
        lay = @layout [a{0.3h} _; b{0.7h, 0.7w} c]

        topfig = histogram(x1_positive, color=positive_color, normalize=true, alpha=0.5, label=nothing, xaxis=nothing)
        histogram!(x1_negative, color=negative_color, normalize=true, alpha=0.5, label=nothing)
        normal_x1_positive = fit_mle(Normal, x1_positive) # NOTE: Try Gamma?
        normal_x1_negative = fit_mle(Normal, x1_negative)
        plot!(x->pdf(normal_x1_positive, x), color=positive_color, xlim=xlims(), label=false, lw=2)
        plot!(x->pdf(normal_x1_negative, x), color=negative_color, xlim=xlims(), label=false, lw=2)
        xlims!(current_xlim) # match limits of main plot

        sidefig = histogram(x2_positive, color=positive_color, normalize=true, alpha=0.5, label=nothing, yaxis=nothing, orientation=:h)
        histogram!(x2_negative, color=negative_color, normalize=true, alpha=0.5, label=nothing, orientation=:h)
        normal_x2_positive = fit_mle(Normal, x2_positive) # NOTE: Try Gamma?
        normal_x2_negative = fit_mle(Normal, x2_negative)
        plot!([pdf(normal_x2_positive, x) for x in pY], pY, color=positive_color, xlim=xlims(), label=false, lw=2)
        plot!([pdf(normal_x2_negative, x) for x in nY], nY, color=negative_color, xlim=xlims(), label=false, lw=2)
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
