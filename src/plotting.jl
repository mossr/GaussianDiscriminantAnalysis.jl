global NEGATIVE_COLOR = "#d62728"
global POSITIVE_COLOR = "#2ca02c"

function plot_gda(𝒟;
                  use_qda=true,
                  soft=true,
                  k=1,
                  show_svm=false,
                  return_predict=false,
                  subplots=false,
                  figsize=[6.4, 4.8],
                  show_legend=false,
                  show_axes=true,
                  flip_colors=true,
                  show_analysis=false)
    # TODO: use Plots interface directly.
    # default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

    if subplots
        figure(figsize=figsize.*1.75)
    else
        figure(figsize=figsize)
    end

    if subplots
        subplot(2,2,3)
    end

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
    dbX = range(min(minimum(x1_positive), minimum(x1_negative)), stop=max(maximum(x1_positive), maximum(x1_negative)), length=1000)
    dbY = range(min(minimum(x2_positive), minimum(x2_negative)) - 2, stop=max(maximum(x2_positive), maximum(x2_negative)), length=1000)
    dbZ = [predict([x,y]) for y in dbY, x in dbX] # Note x-y "for" ordering
    vmin = minimum(dbZ)
    vmax = maximum(dbZ)
    TwoSlopeNorm = PyPlot.matplotlib.colors.TwoSlopeNorm
    if soft
        # Decision boundary is at zero.
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else
        # Decision boundary binary, thus is relative to each k class prediction.
        norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin+vmax)/2, vmax=vmax)
    end
    # Colormap so that when `flip_colors == false`:
    #   red   = class 0 = negative
    #   green = class 1 = positive
    if flip_colors
        cmap_string = "RdYlGn_r"
        positive_color_contour = "plasma"
        negative_color_contour = "viridis"
        positive_color = NEGATIVE_COLOR
        negative_color = POSITIVE_COLOR
    else
        cmap_string = "RdYlGn"
        positive_color_contour = "viridis"
        negative_color_contour = "plasma"
        positive_color = POSITIVE_COLOR
        negative_color = NEGATIVE_COLOR
    end
    contourf(dbX, dbY, dbZ, 100, cmap=cmap_string, vmin=vmin, vmax=vmax, norm=norm)

    #============================================#
    # plot scattered datapoints
    #============================================#
    scatter(x1_negative, x2_negative, label="non-failure", alpha=0.5,color=negative_color, s=10, edgecolor="black")
    scatter(x1_positive, x2_positive, label="failure", alpha=0.5, color=positive_color, s=10, edgecolor="black")

    if !show_axes
        tick_params(bottom=false, labelbottom=false, left=false, labelleft=false)
    end

    # save axis limits
    current_xlim = xlim()
    current_ylim = ylim()

    #============================================#
    # plot multivariate Gaussian contours (positive)
    #============================================#
    fX = range(-2, stop=maximum(x1_positive)*1.1, length=1000)
    fY = range(-2, stop=!use_qda ? maximum(x2_negative)*1.1 : maximum(x2_positive)*1.1 , length=1000)
    fZ = [pdf(mv_positive, [x,y]) for y in fY, x in fX] # Note x-y "for" ordering
    contour(fX, fY, fZ, alpha=0.75, cmap=positive_color_contour)

    #============================================#
    # plot multivariate Gaussian contours (negative)
    #============================================#
    nfX = range(-2, stop=maximum(x1_negative)*1.1, length=1000)
    nfY = range(-2, stop=maximum(x2_negative)*1.1, length=1000)
    nfZ = [pdf(mv_negative, [x,y]) for y in nfY, x in nfX] # Note x-y "for" ordering
    contour(nfX, nfY, nfZ, alpha=0.75, cmap=negative_color_contour)

    # restore axis limits
    xlim(current_xlim)
    ylim(current_ylim)

    # 1D Gaussians
    if subplots
        subplot(4,2,3)
        hist(x1_positive, color=positive_color, density=true, alpha=0.5)
        hist(x1_negative, color=negative_color, density=true, alpha=0.5)
        normal_x1_positive = fit_mle(Normal, x1_positive) # NOTE: Gamma?
        normal_x1_negative = fit_mle(Normal, x1_negative)
        Z_current_xlim = xlim()
        Z_current_ylim = ylim()
        plot(fY, [pdf(normal_x1_positive, x) for x in fY], color=positive_color)
        plot(nfY, [pdf(normal_x1_negative, x) for x in nfY], color=negative_color)
        xlim(Z_current_xlim)
        ylim(Z_current_ylim)
        xticks([])

        subplot(2,4,7)
        hist(x2_positive, orientation="horizontal", color=positive_color, density=true, alpha=0.5)
        hist(x2_negative, orientation="horizontal", color=negative_color, density=true, alpha=0.5)
        normal_x2_positive = fit_mle(Normal, x2_positive) # NOTE: Gamma?
        normal_x2_negative = fit_mle(Normal, x2_negative)
        d_current_xlim = xlim()
        d_current_ylim = ylim()
        base = gca().transData
        rot = matplotlib.transforms.Affine2D().rotate_deg(90)
        plot(fY, [-pdf(normal_x2_positive, x) for x in fY], transform=rot+base, color=positive_color) # notice negative pdf to flip transformation
        plot(nfY, [-pdf(normal_x2_negative, x) for x in nfY], transform=rot+base, color=negative_color) # notice negative pdf to flip transformation
        xlim(d_current_xlim)
        ylim(d_current_ylim)
        yticks([])
    end

    # Boundary line calculated using support vector machines (SVMs)
    if show_svm
        @info "Running SVM..."
        svm_classify, svm_x, svm_y = svm(pos_data, neg_data)
        if subplots
            subplot(2,2,3)
        end
        plot(svm_x, svm_y, label="SVM", color="black")

        if show_analysis
            analyze_fit_svm(svm_classify, pos_data, neg_data)
        end
    end

    if show_analysis
        analyze_fit_gda(predict, pos_data, neg_data)
    end

    if show_legend
        legend()
    end

    # subplots_adjust(wspace=0.08, hspace=0.1)
    tight_layout()
    fig = gcf()

    return return_predict ? (fig, predict, mv_positive, svm_classify) : fig
end