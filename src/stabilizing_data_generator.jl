using FileIO

## stabilizing trajectory data without state estimator

function generate_stabilizing_data(
    model_real, controller, x_min, x_max;
    test_window=1.0, num_train=100, num_test=100, tf=1.0, 
    dt=1/30, resx=50, resy=50, visualize=false, sparse_image=false,
    process_noise_factor=0.0, process_noise_type=:dt,
    input_noise_type=nothing, input_noise_factor=0.0,
    ones_and_zeros=false
)

    println("Sampling initial conditions...")

    # get mean and distance to limits for each state element
    mean_x = 0.5 .* (x_min .+ x_max)
    x_dist = (x_max .- x_min)/2

    # determine indices where mins and maxes are equal (i.e., where the state is fixed)
    x_minmax_eq_ind = findall(x_min .!= x_max)
    
    # sample initial conditions
    x0_training_sampler = Product([Uniform(x_min[i], x_max[i]) for i in x_minmax_eq_ind])
    x0_test_sampler = Product([Uniform(mean_x[i] - (test_window * x_dist[i]), mean_x[i] + (test_window * x_dist[i])) for i in x_minmax_eq_ind])

    # initialize vector of initial conditions
    x0_training = [deepcopy(x_min) for _ in 1:num_train]
    x0_test = [deepcopy(x_min) for _ in 1:num_test]

    # update initial conditions with sampled values at proper indices
    for i in 1:num_train
        x0_training[i][x_minmax_eq_ind] .= rand(x0_training_sampler)
    end

    for i in 1:num_test
        x0_test[i][x_minmax_eq_ind] .= rand(x0_test_sampler)
    end

    # initialize vectors to store trajectory training and test data
    X_pixel_training_traj = sparse_image ? Vector{SparseVector{Float64}}[] : Vector{Vector{Float64}}[]
    X_real_training_traj = Vector{Vector{Float64}}[]
    U_real_training_traj = Vector{Vector{Float64}}[]
    U_est_training_traj = Vector{Vector{Float64}}[]
    T = range(0, tf, step=dt)

    X_pixel_test_traj = sparse_image ? Vector{SparseVector{Float64}}[] : Vector{Vector{Float64}}[]
    X_real_test_traj = Vector{Vector{Float64}}[]
    U_real_test_traj = Vector{Vector{Float64}}[]
    U_est_test_traj = Vector{Vector{Float64}}[]

    println("Generating training data...")

    for x0 in x0_training

        # generate training trajectory using lqr controller based off a nominal model
        X_pixel, X, U, U_est, T = generate_stabilizing_trajectory(
            model_real, controller, x0; 
            tf=tf, dt=dt, resx=resx, resy=resy,
            input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
            process_noise_factor=process_noise_factor, process_noise_type=process_noise_type,
            visualize=visualize, sparse_image=sparse_image,
            ones_and_zeros=ones_and_zeros
        )

        # push trajectory to training data
        push!(X_pixel_training_traj, X_pixel)
        push!(X_real_training_traj, X)
        push!(U_real_training_traj, U)
        push!(U_est_training_traj, U_est)

    end

    println("Generating test data...")

    for x0 in x0_test

        # generate test trajectory using lqr controller based off a nominal model
        X_pixel, X, U, U_est, T = generate_stabilizing_trajectory(
            model_real, controller, x0; 
            tf=tf, dt=dt, resx=resx, resy=resy,
            input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
            process_noise_factor=process_noise_factor, process_noise_type=process_noise_type,
            visualize=visualize, sparse_image=sparse_image,
            ones_and_zeros=ones_and_zeros
        )

        # push trajectory to test data
        push!(X_pixel_test_traj, X_pixel)
        push!(X_real_test_traj, X)
        push!(U_real_test_traj, U)
        push!(U_est_test_traj, U_est)

    end

    println("Done!")

    # combine outputs
    training_data = Dict(
        "X_pixel_training_traj" => X_pixel_training_traj,
        "X_real_training_traj" => X_real_training_traj,
        "U_real_training_traj" => U_real_training_traj,
        "U_est_training_traj" => U_est_training_traj,
        "x0_training" => x0_training,
        "T" => T
    )
    test_data = Dict(
        "X_pixel_test_traj" => X_pixel_test_traj,
        "X_real_test_traj" => X_real_test_traj,
        "U_real_test_traj" => U_real_test_traj,
        "U_est_test_traj" => U_est_test_traj,
        "x0_test" => x0_test,
        "T" => T
    )
    
    return training_data, test_data

end

function generate_stabilizing_trajectory(
    model_real, controller, x0; 
    tf=1.0, dt=1/30, resx=50, resy=50,
    process_noise_factor=0.0, process_noise_type=:dt,
    input_noise_type=nothing, input_noise_factor=0.0,
    visualize=false, sparse_image=false,
    ones_and_zeros=false
)

    X, U, U_est, T = simulatewithcontroller(
        model_real, controller, x0, tf, dt, 
        input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
        process_noise_factor=process_noise_factor, process_noise_type=process_noise_type
    )
    
    # construct a scene for visualization for image generation
    scene = construct_scene(resx, resy)

    # initialize X_pixel
    X_pixel = sparse_image ? SparseVector{Float64}[] : Vector{Float64}[]

    for k in eachindex(X)[1:end]

        # get current state and prev state
        x_k = X[k]

        # visualize and get pixel values
        visualize!(scene, model_real, x_k; visualize=false)
        x_k_pixel = get_scene(scene, sparse_image, ones_and_zeros)

        if visualize
            display(x_k_pixel)
        end

        vec_x_k_pixel = sparse_image ? sparsevec(x_k_pixel) : vec(x_k_pixel)

        push!(X_pixel, vec_x_k_pixel)
    
    end

    # output histories in following format to match dimensions:

    return X_pixel, X, U, U_est, T

end

function generate_stabilizing_data_no_images(
    model_real, controller, x_min, x_max;
    test_window=1.0, num_train=100, num_test=100, tf=1.0, 
    dt=1/30, input_noise_type=:u, input_noise_factor=0.1,
    process_noise_factor=0.0, process_noise_type=:dt
)

    println("Sampling LQR initial conditions...")

    # get mean and distance to limits for each state element
    mean_x = 0.5 .* (x_min .+ x_max)
    x_dist = (x_max .- x_min)/2

    # determine indices where mins and maxes are equal (i.e., where the state is fixed)
    x_minmax_eq_ind = findall(x_min .!= x_max)
    
    # sample initial conditions
    x0_training_sampler = Product([Uniform(x_min[i], x_max[i]) for i in x_minmax_eq_ind])
    x0_test_sampler = Product([Uniform(mean_x[i] - (test_window * x_dist[i]), mean_x[i] + (test_window * x_dist[i])) for i in x_minmax_eq_ind])

    # initialize vector of initial conditions
    x0_training = [deepcopy(x_min) for _ in 1:num_train]
    x0_test = [deepcopy(x_min) for _ in 1:num_test]

    # update initial conditions with sampled values at proper indices
    for i in 1:num_train
        x0_training[i][x_minmax_eq_ind] .= rand(x0_training_sampler)
    end

    for i in 1:num_test
        x0_test[i][x_minmax_eq_ind] .= rand(x0_test_sampler)
    end

    # initialize vectors to store trajectory training and test data
    X_real_training_traj = Vector{Vector{Float64}}[]
    U_real_training_traj = Vector{Vector{Float64}}[]
    U_est_training_traj = Vector{Vector{Float64}}[]
    T = range(0, tf, step=dt)

    X_real_test_traj = Vector{Vector{Float64}}[]
    U_real_test_traj = Vector{Vector{Float64}}[]
    U_est_test_traj = Vector{Vector{Float64}}[]

    println("Generating LQR training data...")

    for x0 in x0_training

        # generate training trajectory using lqr controller based off a nominal model
        X, U, U_est, T = simulatewithcontroller(
            model_real, controller, x0, tf, dt, 
            input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
            process_noise_factor=process_noise_factor, process_noise_type=process_noise_type
        )

        # push trajectory to training data
        push!(X_real_training_traj, X)
        push!(U_real_training_traj, U)
        push!(U_est_training_traj, U_est)

    end

    println("Generating LQR test data...")

    for x0 in x0_test

        # generate test trajectory using lqr controller based off a nominal model
        X, U, U_est, T = simulatewithcontroller(
            model_real, controller, x0, tf, dt, 
            input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
            process_noise_factor=process_noise_factor, process_noise_type=process_noise_type
        )

        # push trajectory to test data
        push!(X_real_test_traj, X)
        push!(U_real_test_traj, U)
        push!(U_est_test_traj, U_est)

    end

    println("Done!")

    # combine outputs
    training_data = [X_real_training_traj, U_real_training_traj, x0_training, T]
    test_data = [X_real_test_traj, U_real_test_traj, x0_test, T]

    training_data = Dict(
        "X_real_training_traj" => X_real_training_traj,
        "U_real_training_traj" => U_real_training_traj,
        "U_est_training_traj" => U_est_training_traj,
        "x0_training" => x0_training,
        "T" => T
    )
    test_data = Dict(
        "X_real_test_traj" => X_real_test_traj,
        "U_real_test_traj" => U_real_test_traj,
        "U_est_test_traj" => U_est_test_traj,
        "x0_test" => x0_test,
        "T" => T
    )

    return training_data, test_data

end

## stabilizing trajectory data with state estimator

function generate_stabilizing_data(
    model_real, controller, estimator, x_min, x_max;
    test_window=1.0, num_train=100, num_test=100, tf=1.0, 
    dt=1/30, resx=50, resy=50, visualize=false, sparse_image=false,
    process_noise_factor=0.0, process_noise_type=:dt,
    input_noise_type=nothing, input_noise_factor=0.0,
    measurement_noise_type=:gaussian, measurement_noise_factor=0.0,
    ones_and_zeros=false
)

    println("Sampling initial conditions...")

    # get mean and distance to limits for each state element
    mean_x = 0.5 .* (x_min .+ x_max)
    x_dist = (x_max .- x_min)/2

    # determine indices where mins and maxes are equal (i.e., where the state is fixed)
    x_minmax_eq_ind = findall(x_min .!= x_max)
    
    # sample initial conditions
    x0_training_sampler = Product([Uniform(x_min[i], x_max[i]) for i in x_minmax_eq_ind])
    x0_test_sampler = Product([Uniform(mean_x[i] - (test_window * x_dist[i]), mean_x[i] + (test_window * x_dist[i])) for i in x_minmax_eq_ind])

    # initialize vector of initial conditions
    x0_training = [deepcopy(x_min) for _ in 1:num_train]
    x0_test = [deepcopy(x_min) for _ in 1:num_test]

    # update initial conditions with sampled values at proper indices
    for i in 1:num_train
        x0_training[i][x_minmax_eq_ind] .= rand(x0_training_sampler)
    end

    for i in 1:num_test
        x0_test[i][x_minmax_eq_ind] .= rand(x0_test_sampler)
    end

    # initialize vectors to store trajectory training and test data
    X_pixel_training_traj = sparse_image ? Vector{SparseVector{Float64}}[] : Vector{Vector{Float64}}[]
    X_real_training_traj = Vector{Vector{Float64}}[]
    X_est_training_traj = Vector{Vector{Float64}}[]
    X_pred_training_traj = Vector{Vector{Float64}}[]
    U_real_training_traj = Vector{Vector{Float64}}[]
    U_est_training_traj = Vector{Vector{Float64}}[]
    Y_training_traj = Vector{Vector{Float64}}[]
    T = range(0, tf, step=dt)

    X_pixel_test_traj = sparse_image ? Vector{SparseVector{Float64}}[] : Vector{Vector{Float64}}[]
    X_real_test_traj = Vector{Vector{Float64}}[]
    X_est_test_traj = Vector{Vector{Float64}}[]
    X_pred_test_traj = Vector{Vector{Float64}}[]
    U_real_test_traj = Vector{Vector{Float64}}[]
    U_est_test_traj = Vector{Vector{Float64}}[]
    Y_test_traj = Vector{Vector{Float64}}[]

    println("Generating training data...")

    for x0 in x0_training

        # generate training trajectory using lqr controller based off a nominal model
        X_real, X_est, X_pred, X_pixel, U_real, U_est, Y, T = generate_stabilizing_trajectory(
            model_real, controller, estimator, x0; 
            tf=tf, dt=dt, resx=resx, resy=resy,
            process_noise_factor=process_noise_factor, process_noise_type=process_noise_type,
            input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
            measurement_noise_type=measurement_noise_type,
            measurement_noise_factor=measurement_noise_factor,
            visualize=visualize, sparse_image=sparse_image,
            ones_and_zeros=ones_and_zeros
        )

        # push trajectory to training data
        push!(X_pixel_training_traj, X_pixel)
        push!(X_real_training_traj, X_real)
        push!(X_est_training_traj, X_est)
        push!(X_pred_training_traj, X_pred)
        push!(U_real_training_traj, U_real)
        push!(U_est_training_traj, U_est)
        push!(Y_training_traj, Y)

    end

    println("Generating test data...")

    for x0 in x0_test

        # generate test trajectory using lqr controller based off a nominal model
        X_real, X_est, X_pred, X_pixel, U_real, U_est, Y, T = generate_stabilizing_trajectory(
            model_real, controller, estimator, x0; 
            tf=tf, dt=dt, resx=resx, resy=resy,
            process_noise_factor=process_noise_factor, process_noise_type=process_noise_type,
            input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
            measurement_noise_type=measurement_noise_type,
            measurement_noise_factor=measurement_noise_factor,
            visualize=visualize, sparse_image=sparse_image,
            ones_and_zeros=ones_and_zeros
        )

        # push trajectory to test data
        push!(X_pixel_test_traj, X_pixel)
        push!(X_real_test_traj, X_real)
        push!(X_est_test_traj, X_est)
        push!(X_pred_test_traj, X_pred)
        push!(U_real_test_traj, U_real)
        push!(U_est_test_traj, U_est)
        push!(Y_test_traj, Y)

    end

    println("Done!")

    # combine outputs
    training_data = Dict(
        "X_pixel_training_traj" => X_pixel_training_traj,
        "X_real_training_traj" => X_real_training_traj,
        "X_est_training_traj" => X_est_training_traj,
        "X_pred_training_traj" => X_pred_training_traj,
        "U_real_training_traj" => U_real_training_traj,
        "U_est_training_traj" => U_est_training_traj,
        "Y_training_traj" => Y_training_traj,
        "x0_training" => x0_training,
        "T" => T
    )
    test_data = Dict(
        "X_pixel_test_traj" => X_pixel_test_traj,
        "X_real_test_traj" => X_real_test_traj,
        "X_est_test_traj" => X_est_test_traj,
        "X_pred_test_traj" => X_pred_test_traj,
        "U_real_test_traj" => U_real_test_traj,
        "U_est_test_traj" => U_est_test_traj,
        "Y_test_traj" => Y_test_traj,
        "x0_test" => x0_test,
        "T" => T
    )
    
    return training_data, test_data

end

function generate_stabilizing_trajectory(
    model_real, controller, estimator, x0; 
    tf=1.0, dt=1/30, resx=50, resy=50,
    process_noise_factor=0.0, process_noise_type=:dt,
    input_noise_type=nothing, input_noise_factor=0.0,
    measurement_noise_type=:gaussian,
    measurement_noise_factor=0.0,
    visualize=false, sparse_image=false,
    ones_and_zeros=false
)

    X_real, X_est, X_pred, U_real, U_est, Y, T = simulatewithcontrollerandestimator(
        model_real, controller, estimator, x0, tf, dt;
        process_noise_factor=process_noise_factor, process_noise_type=process_noise_type,
        input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
        measurement_noise_type=measurement_noise_type, measurement_noise_factor=measurement_noise_factor
    )

    # construct a scene for visualization for image generation
    scene = construct_scene(resx, resy)

    # initialize X_pixel
    X_pixel = sparse_image ? SparseVector{Float64}[] : Vector{Float64}[]

    for k in eachindex(X_real)[1:end]

        # get current state and prev state
        x_k = X_real[k]

        # visualize and get pixel values
        visualize!(scene, model_real, x_k; visualize=false, verbose=true)
        x_k_pixel = get_scene(scene, sparse_image, ones_and_zeros)

        if visualize
            display(x_k_pixel)
        end

        vec_x_k_pixel = sparse_image ? sparsevec(x_k_pixel) : vec(x_k_pixel)

        push!(X_pixel, vec_x_k_pixel)
    
    end

    # output histories in following format to match dimensions:

    return X_real, X_est, X_pred, X_pixel, U_real, U_est, Y, T

end