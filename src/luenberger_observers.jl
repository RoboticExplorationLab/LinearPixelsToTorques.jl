##########################################
## Bilinear Koopman Luenberger observer
##########################################

struct BilinearKoopmanLuenbergerObserver <: AbstractEstimator
    A_hat::AbstractArray
    B_hat::AbstractArray
    C_hat::Vector{<:AbstractArray}
    L::AbstractArray
    d::AbstractArray
    projection::Function
    lifting::Function

end

function BilinearKoopmanLuenbergerObserver(A_hat, B_hat, C_hat, L, d, G,
    function_list, states_list, order_list, scale_list, domain; verbose=false)

    if verbose
        println("Creating Luenberger observer...")
    end

    projection=(z)->G*z
    lifting=(x)->koopman_transform(x, function_list, states_list, order_list; scale_list=scale_list, domain=domain)
    
    BilinearKoopmanLuenbergerObserver(A_hat, B_hat, C_hat, L, d, projection, lifting)

end

function estimator_step(KLO::BilinearKoopmanLuenbergerObserver, xk_corrected, uk, ykp1, k)

    # extract filter parameters
    A_hat = KLO.A_hat
    B_hat = KLO.B_hat
    C_hat = KLO.C_hat
    L = KLO.L
    d = KLO.d
    projection = KLO.projection
    lifting = KLO.lifting

    # project state into Koopman space
    zk_corrected = lifting(xk_corrected)

    # correct and predict Koopman state at next time step k+1
    zkp1_corrected = A_hat*zk_corrected + B_hat*uk + sum(C_hat[i]*zk_corrected .* uk[i] for i in eachindex(uk)) + L*ykp1 + d

    # project Koopman state back into state space
    xkp1_corrected = projection(zkp1_corrected)

    return xkp1_corrected, xkp1_corrected

end

####################################################################
## Learned Luenberger Observer with Learned Feedback LO Dynamics
####################################################################

struct LearnedFeedbackDynamicsLuenbergerObserver <: AbstractEstimator
    A_control_hat::AbstractArray
    L::AbstractArray
    d::AbstractVector

end

function LearnedFeedbackDynamicsLuenbergerObserver(A_control_hat, d, L; verbose=false)

    if verbose
        println("Creating Luenberger observer...")
    end
    
    LearnedFeedbackDynamicsLuenbergerObserver(A_control_hat, L, d)
end

function estimator_step(LO::LearnedFeedbackDynamicsLuenbergerObserver, xk_corrected, uk, ykp1, k)

    # extract filter parameters
    A_control_hat = LO.A_control_hat
    L = LO.L
    d = LO.d

    xkp1_corrected = A_control_hat*xk_corrected + L*ykp1 + d

    return xkp1_corrected, xkp1_corrected

end

########################################
## Kalman Filter
########################################

struct KalmanFilter <: AbstractEstimator
    A::AbstractArray
    B::AbstractArray
    C::AbstractArray
    Q_cov::AbstractArray
    R_cov::AbstractArray
    x_eq::Vector{Float64}
    u_eq::Vector{Float64}
    covariance::Vector{AbstractArray}
    num_y::Int64

end

function KalmanFilter(A::AbstractArray, B, C, Q_cov, R_cov, x_eq, u_eq; verbose=false)

    num_y = size(C, 1)

    if verbose
        println("Creating Kalman filter...")
    end

    # initialize state estimator variables
    covariance = diagm(ones(length(x_eq)))
    
    KalmanFilter(A, B, C, Q_cov, R_cov, Vector(x_eq), Vector(u_eq), [covariance],num_y)

end

function KalmanFilter(model::RobotDynamics.AbstractModel, C, Q_cov, R_cov, x_eq, u_eq, dt; verbose=false)
    
    # if model is bilinear, use discrete jacobian. If not, then assume continuous model and use dfdx and dfdu

    if verbose
        println("Calculating dynamics jacobians...")
    end

    if typeof(model) <: Union{BilinearModel, LinearModel}
        A, B = discrete_jacobian(model, x_eq, u_eq, dt)
    else
        A = dfdx(model, x_eq, u_eq, dt)
        B = dfdu(model, x_eq, u_eq, dt)
    end

    KalmanFilter(A, B, C, Q_cov, R_cov, x_eq, u_eq; verbose=verbose)

end

function reset_covariance!(KF::KalmanFilter)
    KF.covariance[1] .= diagm(ones(length(KF.x_eq)))
    return nothing
end

function update_covariance!(KF::KalmanFilter, cov)
    KF.covariance[1] .= cov
    return nothing
end

function getmeasurement(KF::KalmanFilter, x; measurement_noise_type=:gaussian, measurement_noise_factor=0.0)
    
    y = KF.C*x

    if measurement_noise_type == :gaussian
        y += measurement_noise_factor.*randn(length(y))
    elseif measurement_noise_type == :encoder
        y = measurement_noise_factor .* floor.(y ./ measurement_noise_factor)
    end

    return y

end

function estimator_step(KF::KalmanFilter, xk_corrected, uk, ykp1, k)

    # extract filter parameters
    A = KF.A
    B = KF.B
    C = KF.C
    Q_cov = KF.Q_cov
    R_cov = KF.R_cov
    x_eq = KF.x_eq
    u_eq = KF.u_eq
    covk_corrected = KF.covariance[1]

    ## prediction step

    # predict state at next time step k+1
    xkp1_pred = A*(xk_corrected-x_eq) + B*(uk-u_eq) + x_eq

    # predict measurement at next time step k+1
    ykp1_pred = C*xkp1_pred

    # predict covariance at next time step k+1
    covkp1_pred = A*covk_corrected*A' + Q_cov

    ## correction step

    # Compute Kalman gain
    Pxy = covkp1_pred*C'
    Pyy = C*covkp1_pred*C' + R_cov
    Lk = Pxy * inv(Pyy)

    # Update estimate with measurement
    xkp1_corrected = xkp1_pred + Lk * (ykp1 - ykp1_pred)

    # Update estimate covariance using Joseph form
    update_covariance!(KF, (I - Lk*C) * covkp1_pred * (I - Lk*C)' + Lk*R_cov*Lk')

    return xkp1_corrected, xkp1_pred

end