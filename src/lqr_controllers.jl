#########################################
## Infinite-horizon Stabilizing LQR
#########################################

struct InfLQRController <: AbstractController
    K::Matrix{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    d::Vector{Float64}
    x_eq::Vector{Float64}
    u_eq::Vector{Float64}
    state_error::Function
end

function InfLQRController(A::AbstractArray, B, d, Q, R, x_eq, u_eq;
    verbose=false, state_error=(x,x0)->x-x0, kwargs...
)

    # perform ricatti recursion
    K, = inf_horizon_ricatti(A,B,Q,R; verbose=verbose, kwargs...)

    InfLQRController(Matrix(K), Matrix(A), Matrix(B), Vector(d), Vector(x_eq), Vector(u_eq), state_error)

end

function InfLQRController(model::RobotDynamics.AbstractModel, Q, R, x_eq, u_eq, dt;
    verbose=false, kwargs...
)
    
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

    d = x_eq - A*x_eq - B*u_eq

    InfLQRController(A,B,d,Q,R,x_eq,u_eq; verbose=verbose, kwargs...)

end

function getcontrol(ctrl::InfLQRController, x, k; input_noise_type=nothing, input_noise_factor=0.0, dt=0.01)

    # calculate control input using ricatti K matrix
    dx = ctrl.state_error(x, ctrl.x_eq)
    u = ctrl.u_eq - ctrl.K*dx

    # determine noise
    w_u = zeros(length(u))
    
    if !isnothing(input_noise_type)
        w_u = input_noise_type == :u ? input_noise_factor.*u.*randn(length(u)) : (input_noise_factor/sqrt(dt)) .* randn(length(u))
    end

    u += w_u

    return u

end

function inf_horizon_ricatti(A,B,Q,R; verbose=false, max_iters=1000, tol=1e-6)

    # perform ricatti recursion
    P = Matrix(copy(Q))
    n,m = size(B)
    K = zeros(m,n)
    K_prev = copy(K)

    for k = 1:max_iters

        K .= (R + B'P*B) \ (B'P*A)
        P .= Q + A'P*A - A'P*B*K

        if verbose
            @show norm(K-K_prev,Inf)
        end

        if norm(K-K_prev,Inf) < tol
            verbose && println("Converged in $k iterations")
            return K,P
        end

        K_prev .= K

    end

    @warn "dlqr didn't converge in the given number of iterations ($max_iters)"
    
    return K,P

end