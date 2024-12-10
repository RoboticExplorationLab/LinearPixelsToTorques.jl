###################################
## general simulation functions
###################################

function simulatewithpixelcontrollerandestimator(simulation_model, ctrl, estimator, x0, tf, dt;
    resx=50, resy=50, u0=zeros(RD.control_dim(simulation_model)),
    input_noise_factor=0.0, input_noise_type=nothing,
    process_noise_factor=0.0, process_noise_type=:dt,
    visualize=false, sparse_image=false, verbose=false,
    termination_criteria=nothing, ones_and_zeros=false
)

    # initialize simulation as successful
    successful_sim = true

    # define time range
    T = range(0, tf, step=dt)

    # construct a scene for visualization for image generation
    scene = construct_scene(resx, resy)

    # extract dimensions and trajectory length
    N = length(T)

    # visualize initial state and get greyscale values
    visualize!(scene, simulation_model, x0; visualize=visualize, verbose=verbose)
    yk_pixel_scene = get_scene(scene, sparse_image, ones_and_zeros)
    yk_pixel = sparse_image ? sparsevec(yk_pixel_scene) : vec(yk_pixel_scene)

    # initialize trajectory vectors that will act as list
    X = [copy(x0) for k = 1:N]
    X_est = [copy(x0) for k = 1:N]
    X_pred = [copy(x0) for k = 1:N]

    Y_pixel = [copy(yk_pixel) for k = 1:N]
    U_est = [copy(u0) for k = 1:N-1]
    U = [copy(u0) for k = 1:N-1]

    if verbose
        p = Progress(N-1, 1, "Simulating with pixel ctrl and estimator...")
    end

    # simulate with controller of choice defined by ctrl
    for k = 1:N-1

        try

            # get control from pixel controller
            U_est[k] = getcontrol(ctrl, X_est[k], k; input_noise_factor=input_noise_factor, input_noise_type=input_noise_type, dt=dt)

            if process_noise_type == :u
                U[k] = U_est[k] + process_noise_factor.*U_est[k].*randn(length(U_est[k]))
            elseif process_noise_type == :dt
                U[k] = U_est[k] + (process_noise_factor/sqrt(dt)).*randn(length(U_est[k]))
            else
                U[k] = U_est[k]
            end

            # simulate with simulation model
            if typeof(simulation_model) <: Union{BilinearModel, LinearModel}
                X[k+1] = discrete_dynamics(simulation_model, X[k], U[k], dt)
            else
                X[k+1] = dynamics_rk4(simulation_model, X[k], U[k], dt)
            end

            # visualize and collect pixel values at new x_k+1 as measurements
            visualize!(scene, simulation_model, X[k+1]; visualize=visualize)
            Y_pixel_kp1_scene = get_scene(scene, sparse_image, ones_and_zeros)
            Y_pixel[k+1] = sparse_image ? sparsevec(Y_pixel_kp1_scene) : vec(Y_pixel_kp1_scene)

            X_est[k+1], X_pred[k+1] = estimator_step(estimator, X_est[k], U_est[k], Y_pixel[k+1], k)

            if verbose
                next!(p)
            end

            if termination_criteria != nothing
                if termination_criteria(X[k+1])

                    X = X[1:k]
                    X_est = X_est[1:k]
                    X_pred = X_pred[1:k]
                    U = U[1:k-1]
                    U_est = U_est[1:k-1]
                    Y_pixel = Y_pixel[1:k]
                    T = T[1:k]
                    successful_sim = false

                    break
                end
            end

        catch e
            println("Unstable simulation! Stopping...")
            println(e)

            X = X[1:k]
            X_est = X_est[1:k]
            X_pred = X_pred[1:k]
            U = U[1:k-1]
            U_est = U_est[1:k-1]
            Y_pixel = Y_pixel[1:k]
            T = T[1:k]
            successful_sim = false

            break
        end

    end

    return X, X_est, X_pred, U, U_est, Y_pixel, T, successful_sim

end

function simulatewithcontrollerandestimator(simulation_model, ctrl, estimator, x0, tf, dt;
    u0=zeros(RD.control_dim(simulation_model)), input_noise_factor=0.0, input_noise_type=nothing,
    process_noise_factor=0.0, process_noise_type=:dt,
    measurement_noise_factor=0.0, measurement_noise_type=:gaussian
)

    @assert input_noise_type in [:u, :dt, nothing]
    # define time range
    T = range(0, tf, step=dt)

    # extract dimensions and trajectory length
    N = length(T)

    # initialize trajectory vectors that will act as list
    X = [copy(x0) for k = 1:N]
    X_pred = [copy(x0) for k = 1:N]
    X_est = [copy(x0) for k = 1:N]

    Y = [getmeasurement(estimator, x0;
        measurement_noise_type=measurement_noise_type,
        measurement_noise_factor=measurement_noise_factor) for k = 1:N
    ]
    U = [copy(u0) for k = 1:N-1]
    U_est = [copy(u0) for k = 1:N-1]

    # simulate with controller of choice defined by ctrl
    for k = 1:N-1

        # calculate control input using controller
        U_est[k] = getcontrol(ctrl, X_est[k], k; input_noise_factor=input_noise_factor, input_noise_type=input_noise_type, dt=dt)

        if process_noise_type == :u
            U[k] = U_est[k] + process_noise_factor.*U_est[k].*randn(length(U_est[k]))
        elseif process_noise_type == :dt
            U[k] = U_est[k] + (process_noise_factor/sqrt(dt)).*randn(length(U_est[k]))
        else
            U[k] = U_est[k]
        end

        # rollout a time step to k+1 with "real" dynamics model
        if typeof(simulation_model) <: Union{BilinearModel, LinearModel}
            X[k+1] = discrete_dynamics(simulation_model, X[k], U[k], dt)
        else
            X[k+1] = dynamics_rk4(simulation_model, X[k], U[k], dt)
        end
        
        # collect measurements from sim environment at time k + 1
        Y[k+1] = getmeasurement(estimator, X[k+1];
            measurement_noise_type=measurement_noise_type,
            measurement_noise_factor=measurement_noise_factor
        )

        X_est[k+1], X_pred[k+1] = estimator_step(estimator, X_est[k], U_est[k], Y[k+1], k)

    end

    return X, X_est, X_pred, U, U_est, Y, T

end

function simulatewithcontroller(simulation_model, ctrl, x0, tf, dt;
    u0=zeros(RD.control_dim(simulation_model)),
    process_noise_factor=0.0, process_noise_type=:dt,
    input_noise_factor=0.0, input_noise_type=nothing
)

    @assert input_noise_type in [:u, :dt, nothing]
    # define time range
    T = range(0, tf, step=dt)

    # extract dimensions and trajectory length
    N = length(T)

    # initialize trajectory vectors that will act as list
    X = [copy(x0) for k = 1:N]
    U = [copy(u0) for k = 1:N-1]
    U_est = [copy(u0) for k = 1:N-1]

    # simulate with controller of choice defined by ctrl
    for k = 1:N-1

        U_est[k] = getcontrol(ctrl, X[k], k; input_noise_factor=input_noise_factor, input_noise_type=input_noise_type, dt=dt)

        if process_noise_type == :u
            U[k] = U_est[k] + process_noise_factor.*U_est[k].*randn(length(U_est[k]))
        elseif process_noise_type == :dt
            U[k] = U_est[k] + (process_noise_factor/sqrt(dt)).*randn(length(U_est[k]))
        else
            U[k] = U_est[k]
        end

        if typeof(simulation_model) <: Union{BilinearModel, LinearModel}
            X[k+1] = discrete_dynamics(simulation_model, X[k], U[k], dt)
        else
            X[k+1] = dynamics_rk4(simulation_model, X[k], U[k], dt)
        end

    end

    return X, U, U_est, T

end

function simulatewithestimator(model, estimator, U, x0, tf, dt;
    process_noise_factor=0.0, process_noise_type=:dt,
    measurement_noise_factor=0.0, measurement_noise_type=:gaussian 
)

    # define length of trajectory
    N = round(Int, tf / dt) + 1

    # define time range and initialize state trajectory vector
    T = range(0, tf, length=N)
    X = [copy(x0) for k = 1:N]
    U_real = deepcopy(U)

    if measurement_noise_type == :gaussian
        X_est = [copy(x0) + measurement_noise_factor.*randn(length(x0)) for k = 1:N]
        X_pred = [copy(x0) + measurement_noise_factor.*randn(length(x0)) for k = 1:N]
    elseif measurement_noise_type == :encoder
        x0_est = round.(measurement_noise_factor .* x0) ./ measurement_noise_factor
        X_est = [copy(x0_est) for k = 1:N]
        X_pred = [copy(x0_est) for k = 1:N]
    else
        X_est = [copy(x0) for k = 1:N]
        X_pred = [copy(x0) for k = 1:N]
    end

    Y = [getmeasurement(estimator, x0;
        measurement_noise_type=measurement_noise_type,
        measurement_noise_factor=measurement_noise_factor) for k = 1:N
    ]

    # simulate
    for k = 1:N-1

        if process_noise_type == :u
            U_real[k] += process_noise_factor.*U[k].*randn(length(U[k]))
        elseif process_noise_type == :dt
            U_real[k] +=(process_noise_factor/sqrt(dt)).*randn(length(U[k]))
        end

        if typeof(model) <: Union{BilinearModel, LinearModel}
            X[k+1] = discrete_dynamics(model, X[k], U_real[k], dt)
        else
            X[k+1] = dynamics_rk4(model, X[k], U_real[k], dt)
        end

        # collect measurements from sim environment at time k + 1
        Y[k+1] = getmeasurement(estimator, X[k+1];
            measurement_noise_type=measurement_noise_type,
            measurement_noise_factor=measurement_noise_factor
        )

        X_est[k+1], X_pred[k+1] = estimator_step(estimator, X_est[k], U[k], Y[k+1], k)

    end

    return X, X_est, X_pred, U_real[1:N-1], Y, T

end

function simulate(model, U, x0, tf, dt; process_noise_factor=0.0, process_noise_type=:dt)

    # define length of trajectory
    N = round(Int, tf / dt) + 1

    # define time range and initialize state trajectory vector
    T = range(0, tf, length=N)
    X = [copy(x0) for k = 1:N]
    U_real = deepcopy(U)

    # simulate
    for k = 1:N-1

        if process_noise_type == :u
            U_real[k] += process_noise_factor.*U[k].*randn(length(U[k]))
        elseif process_noise_type == :dt
            U_real[k] +=(process_noise_factor/sqrt(dt)).*randn(length(U[k]))
        end

        if typeof(model) <: Union{BilinearModel, LinearModel}
            X[k+1] = discrete_dynamics(model, X[k], U_real[k], dt)
        else
            X[k+1] = dynamics_rk4(model, X[k], U_real[k], dt)
        end

    end

    return X,U_real[1:N-1],T

end

##############################
## RK4 Dynamics
##############################

function dynamics_rk4(model, x, u, dt)
    #RK4 integration with zero-order hold on u
    f1 = RobotDynamics.dynamics(model, x, u)
    f2 = RobotDynamics.dynamics(model, x + 0.5*dt*f1, u)
    f3 = RobotDynamics.dynamics(model, x + 0.5*dt*f2, u)
    f4 = RobotDynamics.dynamics(model, x + dt*f3, u)
    return x + (dt/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

function dfdx(model, x, u, dt)
    ForwardDiff.jacobian(dx->dynamics_rk4(model, dx, u, dt), x)
end

function dfdu(model, x, u, dt)
    ForwardDiff.jacobian(du->dynamics_rk4(model, x, du, dt), u)
end

function dAdx(model, x, u, dt)
    ForwardDiff.jacobian(dx->vec(dfdx(model, dx, u, dt)), x)
end

function dBdx(model, x, u, dt)
    ForwardDiff.jacobian(dx->dfdu(model, dx, u, dt), x)
end

function dAdu(model, x, u, dt)
    ForwardDiff.jacobian(du->vec(dfdx(model, x, du, dt)), u)
end

function dBdu(model, x, u, dt)
    ForwardDiff.jacobian(du->dfdu(model, x, du, dt), u)
end

function comm(m,n)
    perm = [m*j + i for i in 1:m for j in 0:n-1]
    comm = zeros(m*n,m*n)
    for i = 1:m*n
        comm[i,perm[i]] = 1
    end
    return comm
end