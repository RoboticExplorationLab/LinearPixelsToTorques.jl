
###################################
## general scene functions
###################################

function construct_scene(res)
    scene = Scene(resolution = (res, res), show_axis = false)
    return scene
end

function construct_scene(resx, resy)
    scene = Scene(resolution = (resx, resy), show_axis = false)

    return scene
end

function get_greyscale(scene)
    return convert(Array{Gray{Float64},2}, Makie.colorbuffer(scene))
end

function get_inv_f64(scene)
    gs = get_greyscale(scene)
    f = convert(Array{Float64,2}, gs)
    return 1 .- f
end

function get_ones_and_zeros(scene, sparse_image)
    gs = get_greyscale(scene)
    f = convert(Array{Float64,2}, gs)
    return sparse_image ? f .< 1.0 : f .== 1.0
end

function get_scene(scene, sparse_image::Bool, ones_and_zeros::Bool)
    if ones_and_zeros
        return get_ones_and_zeros(scene, sparse_image)
    else
        return sparse_image ? get_inv_f64(scene) : get_greyscale(scene)
    end
end

function get_rgb(scene)
    return convert(Array{RGB{Float64},2}, Makie.colorbuffer(scene))
end

function get_pixel_state(model, x, res; 
    sparse_image=false, ones_and_zeros=false
)

    scene = construct_scene(res)

    visualize!(scene, model, x)
    x_k_pixel_scene = get_scene(scene, sparse_image, ones_and_zeros)
    x_k_pixel = sparse_image ? sparsevec(x_k_pixel_scene) : vec(x_k_pixel_scene)

    return x_k_pixel

end

function get_pixel_state(model, x, resx, resy; 
    sparse_image=false, ones_and_zeros=false
)

    scene = construct_scene(resx, resy)

    visualize!(scene, model, x)
    x_k_pixel_scene = get_scene(scene, sparse_image, ones_and_zeros)
    x_k_pixel = sparse_image ? sparsevec(x_k_pixel_scene) : vec(x_k_pixel_scene)

    return x_k_pixel

end

function pixel_to_greyscale(x_pixel::AbstractVector, resx, resy)
    return convert(Array{Gray{Float64},2}, reshape(x_pixel, resy, resx))
end

function animate_pixels(model, X_pixel_hist, filename, resx, resy; framerate=30)
    scene = construct_scene(resx, resy)
    X_img = pixel_to_greyscale.(X_pixel_hist, resx, resy)
    frames = 1:length(X_img)

    p = Progress(length(frames), 1, "Creating animation...")

    record(scene, filename, frames, framerate=framerate) do i

        Makie.image!(scene, rotr90(X_img[i]))

        next!(p)

    end
end

function filter_zero_rows_matrix(X)

    # find nonzero rows
    row_idx = findall(vec(sum(abs.(X), dims=2)) .> 0)

    # compute filtering matrix
    P = spzeros(length(row_idx), size(X, 1))
    
    for i in eachindex(row_idx)
        P[i, row_idx[i]] = 1
    end

    return P
end

###################################
## cartpole visualization
###################################

function visualize!(scene::Scene, cartpole::RobotZoo.Cartpole, x; visualize=false, verbose=false)
    
    res = scene.resolution[]
    L = cartpole.l
    pos = x[1]
    θ = -x[2]

    # get cartpole properties
    # get scene limits
    xlim = res[1]/res[2]*10/9 * L
    ylim = 10/9 * L
    if !(-xlim + 1/9*L < pos - L*sin(θ) < xlim - 1/9*L) & verbose
        @warn "Cartpole position out of bounds"
    end
    # @assert xlim/ylim == scene.resolution[][1]/scene.resolution[][2] "ylim is 10/9 * L, adjust scene aspect ratio to match xmax/ylim"
    # empty scene
    empty!(scene)

    # set limits
    scatter!(scene, (-xlim, -ylim), color = :black, markersize = 0)
    scatter!(scene, (xlim, ylim), color = :black, markersize = 0)

    # plot base
    scatter!(scene, (pos, 0.), color = :gray, markersize=res[2]/3, marker = '▬')
    scatter!(scene, (pos, 0.), color=:black, markersize=res[2]/15)

    # plot stick
    px = pos - L * sin(θ)
    py = -L * cos(θ)

    lines!(scene, [(pos, 0.), (px, py)], color=:black, linewidth=res[2]/40)
    
    # plot bob
    scatter!(scene, (px, py), color=:black, markersize=res[2]/15)

    center!(scene)

    if visualize
        display(scene)
    end

    return nothing

end

###################################
## REx cartpole visualization
###################################

function visualize!(scene::Scene, cartpole::RExCartpole, x; visualize=false, verbose=false)

    plot_cartpole = RobotZoo.Cartpole(; mc=cartpole.mc, mp=cartpole.mp, l=cartpole.l, g=cartpole.g)
    x_plot = [x[1], -x[2]+pi, x[3], x[4]]
    visualize!(scene, plot_cartpole, x_plot; visualize=visualize, verbose=verbose)

end