########################################
## Extended Kalman Filter
########################################

struct ExtendedKalmanFilter <: AbstractEstimator
    model::RobotDynamics.AbstractModel
    C::AbstractArray
    Q_cov::AbstractArray
    R_cov::AbstractArray
    covariance::Vector{AbstractArray}
    num_x::Int64
    num_y::Int64
    dt::Float64
    discrete_dynamics::Function
end

function ExtendedKalmanFilter(model, C, Q_cov, R_cov, dt;
    discrete_dynamics=(x,u)->dynamics_rk4(model, x, u, dt), verbose=false)
    
    num_y = size(C, 1)
    num_x = RobotDynamics.state_dim(model)

    if verbose
        println("Creating extended Kalman filter...")
    end

    # initialize state estimator variables
    covariance = diagm(ones(num_x))
    
    ExtendedKalmanFilter(model, C, Q_cov, R_cov, [covariance], num_x, num_y, dt, discrete_dynamics)

end

function reset_covariance!(EKF::ExtendedKalmanFilter)
    EKF.covariance[1] .= diagm(ones(EKF.num_x))
    return nothing
end

function update_covariance!(EKF::ExtendedKalmanFilter, cov)
    EKF.covariance[1] .= cov
    return nothing
end

function getmeasurement(EKF::ExtendedKalmanFilter, x; measurement_noise_type=:gaussian, measurement_noise_factor=0.0)
    
    y = EKF.C*x

    if measurement_noise_type == :gaussian
        y += measurement_noise_factor.*randn(length(y))
    elseif measurement_noise_type == :encoder
        y = measurement_noise_factor .* floor.(y ./ measurement_noise_factor)
    end

    return y

end

function estimator_step(EKF::ExtendedKalmanFilter, xk_corrected, uk, ykp1, k)

    # extract filter parameters
    model = EKF.model
    C = EKF.C
    Q_cov = EKF.Q_cov
    R_cov = EKF.R_cov
    covk_corrected = EKF.covariance[1]
    dt = EKF.dt

    # calculate discrete dynamics jacobian
    if typeof(model) <: BilinearModel
        Ak, _ = discrete_jacobian(model, xk_corrected, uk, dt)
    else
        Ak = dfdx(model, xk_corrected, uk, dt)
    end

    ## prediction step

    # predict state at next time step k+1
    xkp1_pred = EKF.discrete_dynamics(xk_corrected, uk)

    # predict measurement at next time step k+1
    ykp1_pred = C*xkp1_pred

    # predict covariance at next time step k+1
    covkp1_pred = Ak*covk_corrected*Ak' + Q_cov

    ## correction step

    # Compute Kalman gain
    Pxy = covkp1_pred*C'
    Pyy = C*covkp1_pred*C' + R_cov
    Lk = Pxy * inv(Pyy)

    # Update estimate with measurement
    xkp1_corrected = xkp1_pred + Lk * (ykp1 - ykp1_pred)

    # Update estimate covariance using Joseph form
    update_covariance!(EKF, (I - Lk*C) * covkp1_pred * (I - Lk*C)' + Lk*R_cov*Lk')

    return xkp1_corrected, xkp1_pred

end