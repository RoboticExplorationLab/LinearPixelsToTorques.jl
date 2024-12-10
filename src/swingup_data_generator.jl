using FileIO

## swingup trajectory data without state estimator

function generate_swingup_data(
    model_real, traj_generator, controller, x_goal, u_goal,
    x_min, x_max, u_min, u_max; max_iter=5000, test_window=1.0, num_train_ref=10,
    num_test_ref=10, num_rollouts_per_ref=10, tf=1.0, dt=1/30, resx=50, resy=50, visualize=false,
    sparse_image=false, input_noise_type=:dt, input_noise_factor=0.0,
    process_noise_factor=0.0, process_noise_type=:dt,
    ones_and_zeros=false
)

    Random.seed!(1)

    println("Sampling initial conditions...")

    # get mean and distance to limits for each state and control element
    mean_x = 0.5 .* (x_min .+ x_max)
    x_dist = (x_max .- x_min)/2

    mean_u = 0.5 .* (u_min .+ u_max)
    u_dist = (u_max .- u_min)/2

    # determine indices where mins and maxes are equal (i.e., where the state and controls are fixed)
    x_minmax_eq_ind = findall(x_min .!= x_max)
    u_minmax_eq_ind = findall(u_min .!= u_max)
    
    # sample initial conditions
    x0_training_sampler = Product([Uniform(x_min[i], x_max[i]) for i in x_minmax_eq_ind])
    x0_test_sampler = Product([Uniform(mean_x[i] - (test_window * x_dist[i]), mean_x[i] + (test_window * x_dist[i])) for i in x_minmax_eq_ind])

    u0_training_sampler = Product([Uniform(u_min[i], u_max[i]) for i in u_minmax_eq_ind])
    u0_test_sampler = Product([Uniform(mean_u[i] - (test_window * u_dist[i]), mean_u[i] + (test_window * u_dist[i])) for i in u_minmax_eq_ind])

    # initialize vector of initial conditions
    x0_ref = [deepcopy(x_min) for _ in 1:num_train_ref]
    x0_test = [deepcopy(x_min) for _ in 1:num_test_ref]

    u0_ref = [deepcopy(u_min) for _ in 1:num_train_ref]
    u0_test = [deepcopy(u_min) for _ in 1:num_test_ref]

    # update initial conditions with sampled values at proper indices
    for i in 1:num_train_ref
        x0_ref[i][x_minmax_eq_ind] .= rand(x0_training_sampler)
        u0_ref[i][u_minmax_eq_ind] .= rand(u0_training_sampler)
    end

    for i in 1:num_test_ref
        x0_test[i][x_minmax_eq_ind] .= rand(x0_test_sampler)
        u0_test[i][u_minmax_eq_ind] .= rand(u0_test_sampler)
    end

    # # iterate through each edge case of initial conditions
    # iter = 0
    # for combo_x in subsets(x_minmax_eq_ind)
    #     for combo_u in subsets(u_minmax_eq_ind)
    #         iter += 1

    #         if iter > num_train_ref
    #             break
    #         end
            
    #         if !isempty(combo_x)
    #             x0_ref[iter][combo_x] .= x_min[combo_x]
    #         end

    #         if !isempty(combo_u)
    #             u0_ref[iter][combo_u] .= u_min[combo_u]
    #         end

    #         if !isempty(setdiff(x_minmax_eq_ind, combo_x))
    #             x0_ref[iter][setdiff(x_minmax_eq_ind, combo_x)] .= x_max[setdiff(x_minmax_eq_ind, combo_x)]
    #         end

    #         if !isempty(setdiff(u_minmax_eq_ind, combo_u))
    #             u0_ref[iter][setdiff(u_minmax_eq_ind, combo_u)] .= u_max[setdiff(u_minmax_eq_ind, combo_u)]
    #         end

    #     end
    # end

    # initialize vectors to store trajectory training and test data
    X_pixel_training_traj = sparse_image ? Vector{SparseVector{Float64}}[] : Vector{Vector{Float64}}[]
    X_real_training_traj = Vector{Vector{Float64}}[]
    X_ref_training_traj = Vector{Vector{Float64}}[]
    U_real_training_traj = Vector{Vector{Float64}}[]
    U_est_training_traj = Vector{Vector{Float64}}[]
    U_ref_training_traj = Vector{Vector{Float64}}[]
    T = range(0, tf, step=dt)

    X_pixel_test_traj = sparse_image ? Vector{SparseVector{Float64}}[] : Vector{Vector{Float64}}[]
    X_real_test_traj = Vector{Vector{Float64}}[]
    X_ref_test_traj = Vector{Vector{Float64}}[]
    U_real_test_traj = Vector{Vector{Float64}}[]
    U_est_test_traj = Vector{Vector{Float64}}[]
    U_ref_test_traj = Vector{Vector{Float64}}[]

    println("Generating training data...")

    for i in eachindex(x0_ref)

        x0 = x0_ref[i]
        u0 = u0_ref[i]

        # generate reference trajectories for tracking
        X_ref, U_ref, _ = generate_trajectory(traj_generator, x0, u0, x_goal, u_goal, dt, tf; max_iter=max_iter)

        # run multiple rollouts of the reference trajectory to generate training data
        for _ in 1:num_rollouts_per_ref

            if typeof(controller) <: TVLQRController
                update_controller!(controller, X_ref, U_ref, dt)
            end

            # generate training trajectory using lqr controller based off a nominal model
            X_pixel, X, U, U_est, T = generate_swingup_trajectory(
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
            push!(X_ref_training_traj, X_ref)
            push!(U_est_training_traj, U_est)
            push!(U_ref_training_traj, U_ref)

        end

    end

    println("Generating test data...")

    for i in eachindex(x0_test)

        x0 = x0_test[i]
        u0 = u0_test[i]

        # generate reference trajectories for tracking
        X_ref, U_ref, _ = generate_trajectory(traj_generator, x0, u0, x_goal, u_goal, dt, tf; max_iter=max_iter)

        for _ in 1:num_rollouts_per_ref

            if typeof(controller) <: TVLQRController
                update_controller!(controller, X_ref, U_ref, dt)
            end

            # generate test trajectory using lqr controller based off a nominal model
            X_pixel, X, U, U_est, T = generate_swingup_trajectory(
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
            push!(X_ref_test_traj, X_ref)
            push!(U_ref_test_traj, U_ref)
            push!(U_est_test_traj, U_est)

        end

    end

    println("Done!")

    # combine outputs
    training_data = Dict(
        "X_pixel_training_traj" => X_pixel_training_traj,
        "X_real_training_traj" => X_real_training_traj,
        "X_ref_training_traj" => X_ref_training_traj,
        "U_real_training_traj" => U_real_training_traj,
        "U_est_training_traj" => U_est_training_traj,
        "U_ref_training_traj" => U_ref_training_traj,
        "x0_ref" => x0_ref,
        "u0_ref" => u0_ref,
        "T" => T
    )
    test_data = Dict(
        "X_pixel_test_traj" => X_pixel_test_traj,
        "X_real_test_traj" => X_real_test_traj,
        "X_ref_test_traj" => X_ref_test_traj,
        "U_real_test_traj" => U_real_test_traj,
        "U_est_test_traj" => U_est_test_traj,
        "U_ref_test_traj" => U_ref_test_traj,
        "x0_test" => x0_test,
        "u0_test" => u0_test,
        "T" => T
    )

    return training_data, test_data

end

function generate_swingup_trajectory(
    model_real, controller, x0; 
    tf=1.0, dt=1/30, resx=50, resy=50,
    input_noise_type=:dt, input_noise_factor=0.0,
    process_noise_factor=0.0, process_noise_type=:dt,
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

function generate_swingup_data_no_images(
    model_real, controller, x_min, x_max;
    max_iter=5000, test_window=1.0, num_train_ref=10,
    num_test_ref=10, num_rollouts_per_ref=10, tf=1.0, 
    dt=1/30, input_noise_type=:u, input_noise_factor=0.1,
    process_noise_factor=0.0, process_noise_type=:dt
)

    Random.seed!(1)

    println("Sampling initial conditions...")

    # get mean and distance to limits for each state and control element
    mean_x = 0.5 .* (x_min .+ x_max)
    x_dist = (x_max .- x_min)/2

    mean_u = 0.5 .* (u_min .+ u_max)
    u_dist = (u_max .- u_min)/2

    # determine indices where mins and maxes are equal (i.e., where the state and controls are fixed)
    x_minmax_eq_ind = findall(x_min .!= x_max)
    u_minmax_eq_ind = findall(u_min .!= u_max)

    # sample initial conditions
    x0_training_sampler = Product([Uniform(x_min[i], x_max[i]) for i in x_minmax_eq_ind])
    x0_test_sampler = Product([Uniform(mean_x[i] - (test_window * x_dist[i]), mean_x[i] + (test_window * x_dist[i])) for i in x_minmax_eq_ind])

    u0_training_sampler = Product([Uniform(u_min[i], u_max[i]) for i in u_minmax_eq_ind])
    u0_test_sampler = Product([Uniform(mean_u[i] - (test_window * u_dist[i]), mean_u[i] + (test_window * u_dist[i])) for i in u_minmax_eq_ind])

    # initialize vector of initial conditions
    x0_ref = [deepcopy(x_min) for _ in 1:num_train_ref]
    x0_test = [deepcopy(x_min) for _ in 1:num_test_ref]

    u0_ref = [deepcopy(u_min) for _ in 1:num_train_ref]
    u0_test = [deepcopy(u_min) for _ in 1:num_test_ref]

    # update initial conditions with sampled values at proper indices
    for i in 1:num_train_ref
        x0_ref[i][x_minmax_eq_ind] .= rand(x0_training_sampler)
        u0_ref[i][u_minmax_eq_ind] .= rand(u0_training_sampler)
    end

    for i in 1:num_test_ref
        x0_test[i][x_minmax_eq_ind] .= rand(x0_test_sampler)
        u0_test[i][u_minmax_eq_ind] .= rand(u0_test_sampler)
    end

    # # iterate through each edge case of initial conditions
    # iter = 0
    # for combo_x in subsets(x_minmax_eq_ind)
    #     for combo_u in subsets(u_minmax_eq_ind)
    #         iter += 1

    #         if iter > num_train_ref
    #             break
    #         end
            
    #         if !isempty(combo_x)
    #             x0_ref[iter][combo_x] .= x_min[combo_x]
    #         end

    #         if !isempty(combo_u)
    #             u0_ref[iter][combo_u] .= u_min[combo_u]
    #         end

    #         if !isempty(setdiff(x_minmax_eq_ind, combo_x))
    #             x0_ref[iter][setdiff(x_minmax_eq_ind, combo_x)] .= x_max[setdiff(x_minmax_eq_ind, combo_x)]
    #         end

    #         if !isempty(setdiff(u_minmax_eq_ind, combo_u))
    #             u0_ref[iter][setdiff(u_minmax_eq_ind, combo_u)] .= u_max[setdiff(u_minmax_eq_ind, combo_u)]
    #         end

    #     end
    # end

    # initialize vectors to store trajectory training and test data
    X_real_training_traj = Vector{Vector{Float64}}[]
    X_ref_training_traj = Vector{Vector{Float64}}[]
    U_real_training_traj = Vector{Vector{Float64}}[]
    U_est_training_traj = Vector{Vector{Float64}}[]
    U_ref_training_traj = Vector{Vector{Float64}}[]
    T = range(0, tf, step=dt)

    X_real_test_traj = Vector{Vector{Float64}}[]
    X_ref_test_traj = Vector{Vector{Float64}}[]
    U_real_test_traj = Vector{Vector{Float64}}[]
    U_est_test_traj = Vector{Vector{Float64}}[]
    U_ref_test_traj = Vector{Vector{Float64}}[]

    println("Generating training data...")

    for i in eachindex(x0_ref)

        x0 = x0_ref[i]
        u0 = u0_ref[i]

        # generate reference trajectories for tracking
        X_ref, U_ref, _ = generate_trajectory(traj_generator, x0, u0, x_goal, u_goal, dt, tf; max_iter=max_iter)

        # push reference trajectory to data
        push!(X_ref_training_traj, X_ref)
        push!(U_ref_training_traj, U_ref)

        # run multiple rollouts of the reference trajectory to generate training data
        for _ in 1:num_rollouts_per_ref

            if typeof(controller) <: TVLQRController
                update_controller!(controller, X_ref, U_ref, dt)
            end

            # generate training trajectory using lqr controller based off a nominal model
            X, U, U_est, _ = simulatewithcontroller(
                model_real, controller, x0, tf, dt, 
                input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
                process_noise_factor=process_noise_factor, process_noise_type=process_noise_type
            )

            # push trajectory to training data
            push!(X_real_training_traj, X)
            push!(U_real_training_traj, U)
            push!(U_est_training_traj, U_est)

        end

    end

    println("Generating test data...")

    for i in eachindex(x0_test)

        x0 = x0_test[i]
        u0 = u0_test[i]

        # generate reference trajectories for tracking
        X_ref, U_ref, _ = generate_trajectory(traj_generator, x0, u0, x_goal, u_goal, dt, tf; max_iter=max_iter)

        # push reference trajectory to data
        push!(X_ref_test_traj, X_ref)
        push!(U_ref_test_traj, U_ref)

        for _ in 1:num_rollouts_per_ref

            if typeof(controller) <: TVLQRController
                update_controller!(controller, X_ref, U_ref, dt)
            end

            # generate test trajectory using lqr controller based off a nominal model
            X, U, U_est, _ = simulatewithcontroller(
                model_real, controller, x0, tf, dt, 
                input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
                process_noise_factor=process_noise_factor, process_noise_type=process_noise_type
            )

            # push trajectory to test data
            push!(X_real_test_traj, X)
            push!(U_real_test_traj, U)
            push!(U_est_test_traj, U_est)

        end

    end

    println("Done!")

    # combine outputs
    training_data = Dict(
        "X_real_training_traj" => X_real_training_traj,
        "X_ref_training_traj" => X_ref_training_traj,
        "U_real_training_traj" => U_real_training_traj,
        "U_est_training_traj" => U_est_training_traj,
        "U_ref_training_traj" => U_ref_training_traj,
        "x0_ref" => x0_ref,
        "u0_ref" => u0_ref,
        "T" => T
    )
    test_data = Dict(
        "X_real_test_traj" => X_real_test_traj,
        "X_ref_test_traj" => X_ref_test_traj,
        "U_real_test_traj" => U_real_test_traj,
        "U_est_test_traj" => U_est_test_traj,
        "U_ref_test_traj" => U_ref_test_traj,
        "x0_test" => x0_test,
        "u0_test" => u0_test,
        "T" => T
    )

    return training_data, test_data
end

## swingup trajectory data with state estimator

function generate_swingup_data(
    model_real, traj_generator, controller, estimator, x_goal, u_goal,
    x_min, x_max, u_min, u_max, x0_perturb; max_iter=5000, test_window=1.0, num_train_ref=10,
    num_test_ref=10, num_rollouts_per_ref=10, tf=1.0, 
    dt=1/30, resx=50, resy=50, visualize=false, sparse_image=false, 
    process_noise_factor=0.0, process_noise_type=:dt,
    input_noise_type=:dt, input_noise_factor=0.0,
    measurement_noise_type=:gaussian, measurement_noise_factor=0.0,
    ones_and_zeros=false
)

    Random.seed!(1)

    println("Sampling initial conditions...")

    # get mean and distance to limits for each state and control element
    mean_x = 0.5 .* (x_min .+ x_max)
    x_dist = (x_max .- x_min)/2

    mean_u = 0.5 .* (u_min .+ u_max)
    u_dist = (u_max .- u_min)/2

    # determine indices where mins and maxes are equal (i.e., where the state and controls are fixed)
    x_minmax_eq_ind = findall(x_min .!= x_max)
    u_minmax_eq_ind = findall(u_min .!= u_max)
    x0_minmax_eq_ind = findall(x0_perturb .!= 0.0)

    # sample initial conditions
    x0_training_sampler = Product([Uniform(x_min[i], x_max[i]) for i in x_minmax_eq_ind])
    x0_test_sampler = Product([Uniform(mean_x[i] - (test_window * x_dist[i]), mean_x[i] + (test_window * x_dist[i])) for i in x_minmax_eq_ind])

    u0_training_sampler = Product([Uniform(u_min[i], u_max[i]) for i in u_minmax_eq_ind])
    u0_test_sampler = Product([Uniform(mean_u[i] - (test_window * u_dist[i]), mean_u[i] + (test_window * u_dist[i])) for i in u_minmax_eq_ind])
    
    # initialize vector of initial conditions
    x0_ref_training = [deepcopy(x_min) for _ in 1:num_train_ref]
    x0_ref_test = [deepcopy(x_min) for _ in 1:num_test_ref]

    u0_ref_training = [deepcopy(u_min) for _ in 1:num_train_ref]
    u0_ref_test = [deepcopy(u_min) for _ in 1:num_test_ref]

    # update initial conditions with sampled values at proper indices
    for i in 1:num_train_ref
        x0_ref_training[i][x_minmax_eq_ind] .= rand(x0_training_sampler)
        u0_ref_training[i][u_minmax_eq_ind] .= rand(u0_training_sampler)
    end

    for i in 1:num_test_ref
        x0_ref_test[i][x_minmax_eq_ind] .= rand(x0_test_sampler)
        u0_ref_test[i][u_minmax_eq_ind] .= rand(u0_test_sampler)
    end

    # iterate through each edge case of initial conditions
    # iter = 0
    # for combo_x in subsets(x_minmax_eq_ind)
    #     for combo_u in subsets(u_minmax_eq_ind)
    #         iter += 1

    #         if iter > num_train_ref
    #             break
    #         end
            
    #         if !isempty(combo_x)
    #             x0_ref[iter][combo_x] .= x_min[combo_x]
    #         end

    #         if !isempty(combo_u)
    #             u0_ref[iter][combo_u] .= u_min[combo_u]
    #         end

    #         if !isempty(setdiff(x_minmax_eq_ind, combo_x))
    #             x0_ref[iter][setdiff(x_minmax_eq_ind, combo_x)] .= x_max[setdiff(x_minmax_eq_ind, combo_x)]
    #         end

    #         if !isempty(setdiff(u_minmax_eq_ind, combo_u))
    #             u0_ref[iter][setdiff(u_minmax_eq_ind, combo_u)] .= u_max[setdiff(u_minmax_eq_ind, combo_u)]
    #         end

    #     end
    # end

    # initialize vectors to store trajectory training and test data
    X_pixel_training_traj = sparse_image ? Vector{SparseVector{Float64}}[] : Vector{Vector{Float64}}[]
    X_real_training_traj = Vector{Vector{Float64}}[]
    X_est_training_traj = Vector{Vector{Float64}}[]
    X_pred_training_traj = Vector{Vector{Float64}}[]
    X_ref_training_traj = Vector{Vector{Float64}}[]
    U_real_training_traj = Vector{Vector{Float64}}[]
    U_est_training_traj = Vector{Vector{Float64}}[]
    U_ref_training_traj = Vector{Vector{Float64}}[]
    Y_training_traj = Vector{Vector{Float64}}[]
    x0_training = Vector{Float64}[]
    T = range(0, tf, step=dt)

    X_pixel_test_traj = sparse_image ? Vector{SparseVector{Float64}}[] : Vector{Vector{Float64}}[]
    X_real_test_traj = Vector{Vector{Float64}}[]
    X_est_test_traj = Vector{Vector{Float64}}[]
    X_pred_test_traj = Vector{Vector{Float64}}[]
    X_ref_test_traj = Vector{Vector{Float64}}[]
    U_real_test_traj = Vector{Vector{Float64}}[]
    U_est_test_traj = Vector{Vector{Float64}}[]
    U_ref_test_traj = Vector{Vector{Float64}}[]
    Y_test_traj = Vector{Vector{Float64}}[]
    x0_test = Vector{Float64}[]

    println("Generating training data...")

    for i in 1:num_train_ref

        x0_ref = x0_ref_training[i]
        u0_ref = u0_ref_training[i]

        # generate reference trajectories for tracking
        X_ref, U_ref, _ = generate_trajectory(traj_generator, x0_ref, u0_ref, x_goal, u_goal, dt, tf; max_iter=max_iter)

        # sample initial conditions for tracking about X_ref, U_ref
        x0_tracking_sample = Product([Uniform(x0_ref[i] - x0_perturb[i], x0_ref[i] + x0_perturb[i]) for i in x0_minmax_eq_ind])

        # run multiple rollouts of the reference trajectory to generate training data
        for _ in 1:num_rollouts_per_ref

            x0_tracking = deepcopy(x0_ref)
            x0_tracking[x0_minmax_eq_ind] .= rand(x0_tracking_sample)

            if typeof(controller) <: TVLQRController
                update_controller!(controller, X_ref, U_ref, dt)
            end

            # generate training trajectory using lqr controller based off a nominal model
            X_real, X_est, X_pred, X_pixel, U_real, U_est, Y, T= generate_swingup_trajectory(
                model_real, controller, estimator, x0_tracking;
                tf=tf, dt=dt, resx=resx, resy=resy,
                process_noise_factor=process_noise_factor, process_noise_type=process_noise_type,
                input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
                measurement_noise_type=measurement_noise_type, measurement_noise_factor=measurement_noise_factor,
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
            push!(X_ref_training_traj, X_ref)
            push!(U_ref_training_traj, U_ref)
            push!(x0_training, x0_tracking)

        end

    end

    println("Generating test data...")

    if num_test_ref == 0

        for i in 1:num_train_ref

            x0_ref = x0_ref_training[i]
            u0_ref = u0_ref_training[i]

            # generate reference trajectories for tracking
            X_ref, U_ref, _ = generate_trajectory(traj_generator, x0_ref, u0_ref, x_goal, u_goal, dt, tf; max_iter=max_iter)

            # sample initial conditions for tracking about X_ref, U_ref
            x0_tracking_sample = Product([Uniform(X_ref[1][i] - test_window.*x0_perturb[i], X_ref[1][i] + test_window.*x0_perturb[i]) for i in x0_minmax_eq_ind])

            for _ in 1:num_rollouts_per_ref

                x0_tracking = deepcopy(X_ref[1])
                x0_tracking[x0_minmax_eq_ind] .= rand(x0_tracking_sample)

                if typeof(controller) <: TVLQRController
                    update_controller!(controller, X_ref, U_ref, dt)
                end

                # generate test trajectory using lqr controller based off a nominal model
                X_real, X_est, X_pred, X_pixel, U_real, U_est, Y, T = generate_swingup_trajectory(
                    model_real, controller, estimator, x0_tracking;
                    tf=tf, dt=dt, resx=resx, resy=resy,
                    process_noise_factor=process_noise_factor, process_noise_type=process_noise_type,
                    input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
                    measurement_noise_type=measurement_noise_type, measurement_noise_factor=measurement_noise_factor,
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
                push!(X_ref_test_traj, X_ref)
                push!(U_ref_test_traj, U_ref)
                push!(x0_test, x0_tracking)

            end

        end

    else

        for i in eachindex(x0_test)

            x0_ref = x0_ref_test[i]
            u0_ref = u0_ref_test[i]

            # generate reference trajectories for tracking
            X_ref, U_ref, _ = generate_trajectory(traj_generator, x0_ref, u0_ref, x_goal, u_goal, dt, tf; max_iter=max_iter)

            # sample initial conditions for tracking about X_ref, U_ref
            x0_tracking_sample = Product([Uniform(x0_ref[i] - x0_perturb[i], x0_ref[i] + x0_perturb[i]) for i in x0_minmax_eq_ind])

            for _ in 1:num_rollouts_per_ref

                x0_tracking = deepcopy(x0_ref)
                x0_tracking[x0_minmax_eq_ind] = rand(x0_tracking_sample)

                if typeof(controller) <: TVLQRController
                    update_controller!(controller, X_ref, U_ref, dt)
                end
                
                # generate test trajectory using lqr controller based off a nominal model
                X_real, X_est, X_pred, X_pixel, U_real, U_est, Y, T = generate_swingup_trajectory(
                    model_real, controller, estimator, x0_tracking;
                    tf=tf, dt=dt, resx=resx, resy=resy,
                    process_noise_factor=process_noise_factor, process_noise_type=process_noise_type,
                    input_noise_type=input_noise_type, input_noise_factor=input_noise_factor,
                    measurement_noise_type=measurement_noise_type, measurement_noise_factor=measurement_noise_factor,
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
                push!(X_ref_test_traj, X_ref)
                push!(U_ref_test_traj, U_ref)
                push!(x0_test, x0_tracking)

            end

        end

    end

    println("Done!")

    # combine outputs
    training_data = Dict(
        "X_pixel_training_traj" => X_pixel_training_traj,
        "X_real_training_traj" => X_real_training_traj,
        "X_est_training_traj" => X_est_training_traj,
        "X_pred_training_traj" => X_pred_training_traj,
        "X_ref_training_traj" => X_ref_training_traj,
        "U_real_training_traj" => U_real_training_traj,
        "U_est_training_traj" => U_est_training_traj,
        "U_ref_training_traj" => U_ref_training_traj,
        "Y_training_traj" => Y_training_traj,
        "x0_training" => x0_training,
        "T" => T
    )
    test_data = Dict(
        "X_pixel_test_traj" => X_pixel_test_traj,
        "X_real_test_traj" => X_real_test_traj,
        "X_est_test_traj" => X_est_test_traj,
        "X_pred_test_traj" => X_pred_test_traj,
        "X_ref_test_traj" => X_ref_test_traj,
        "U_real_test_traj" => U_real_test_traj,
        "U_est_test_traj" => U_est_test_traj,
        "U_ref_test_traj" => U_ref_test_traj,
        "Y_test_traj" => Y_test_traj,
        "x0_test" => x0_test,
        "T" => T
    )

    return training_data, test_data

end

function generate_swingup_trajectory(
    model_real, controller, estimator, x0; 
    tf=1.0, dt=1/30, resx=50, resy=50,
    process_noise_factor=0.0, process_noise_type=:dt,
    input_noise_type=:dt, input_noise_factor=0.0,
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
        visualize!(scene, model_real, x_k; visualize=false)
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