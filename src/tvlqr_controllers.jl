########################################
## Finite-horizon Stabilizing LQR
########################################

struct TVLQRController <: AbstractController
    model::RobotDynamics.AbstractModel
    A_ref::Vector{Vector{Matrix{Float64}}}
    B_ref::Vector{Vector{Matrix{Float64}}}
    Q::Matrix{Float64}
    Qf::Matrix{Float64}
    R::Matrix{Float64}
    K::Vector{Vector{Matrix{Float64}}}
    X_ref::Vector{Vector{Vector{Float64}}}
    U_ref::Vector{Vector{Vector{Float64}}}
    N::Int64
    state_error::Function
end

function TVLQRController(model::RobotDynamics.AbstractModel, A_ref::AbstractArray{<:AbstractArray}, B_ref, Q, Qf, R, X_ref, U_ref, N;
    verbose=false, state_error=(x,x0)->x-x0, kwargs...
)
    
    K, _ = finite_horizon_ricatti(A_ref,B_ref,Q,Qf,R,N; verbose=verbose, kwargs...)

    TVLQRController(model, [A_ref], [B_ref], Matrix(Q), Matrix(Qf), Matrix(R), [K], [X_ref], [U_ref], Int(N), state_error)

end

function TVLQRController(model::RobotDynamics.AbstractModel, Q, Qf, R, X_ref, U_ref, dt, N;
    verbose=false, kwargs...
)
    
    # if model is bilinear, use discrete jacobian. If not, then assume continuous model and use dfdx and dfdu

    if verbose
        println("Calculating dynamics jacobians...")
    end

    A_ref = Vector{Matrix{Float64}}(undef, length(X_ref))
    B_ref = Vector{Matrix{Float64}}(undef, length(U_ref))

    for k in eachindex(U_ref)
        
        if typeof(model) <: Union{BilinearModel}
            Ak, Bk = discrete_jacobian(model, X_ref[k], U_ref[k], dt)
        else
            Ak = dfdx(model, X_ref[k], U_ref[k], dt)
            Bk = dfdu(model, X_ref[k], U_ref[k], dt)
        end
    
        A_ref[k] = Ak
        B_ref[k] = Bk
    
    end

    if typeof(model) <: Union{BilinearModel}
        An, _ = discrete_jacobian(model, X_ref[end], zeros(length(U_ref[1])), dt)
    else
        An = dfdx(model, X_ref[end], zeros(length(U_ref[1])), dt)
    end
    A_ref[end] = An

    TVLQRController(model, A_ref,B_ref,Q,Qf,R,X_ref,U_ref,N; verbose=verbose, kwargs...)

end

function TVLQRController(model::RobotDynamics.AbstractModel, Q, Qf, R, dt, N;
    verbose=false, kwargs...
)

    X_ref = [zeros(size(Q, 2)) for _ = 1:N]
    U_ref = [zeros(size(R, 2)) for _ = 1:N-1]

    TVLQRController(model, Q, Qf, R, X_ref, U_ref, dt, N; verbose=verbose, kwargs...)

end

function update_controller!(ctrl::TVLQRController, X_ref, U_ref, dt)

    model = ctrl.model

    A_ref = Vector{Matrix{Float64}}(undef, length(X_ref))
    B_ref = Vector{Matrix{Float64}}(undef, length(U_ref))

    for k in eachindex(U_ref)
        
        if typeof(model) <: Union{BilinearModel}
            Ak, Bk = discrete_jacobian(model, X_ref[k], U_ref[k], dt)
        else
            Ak = dfdx(model, X_ref[k], U_ref[k], dt)
            Bk = dfdu(model, X_ref[k], U_ref[k], dt)
        end
    
        A_ref[k] = Ak
        B_ref[k] = Bk
    
    end

    if typeof(model) <: Union{BilinearModel}
        An, _ = discrete_jacobian(model, X_ref[end], zeros(length(U_ref[1])), dt)
    else
        An = dfdx(model, X_ref[end], zeros(length(U_ref[1])), dt)
    end
    A_ref[end] = An

    K, _ = finite_horizon_ricatti(A_ref,B_ref,ctrl.Q,ctrl.Qf,ctrl.R,ctrl.N)

    ctrl.A_ref[1] .= A_ref
    ctrl.B_ref[1] .= B_ref
    ctrl.X_ref[1] .= X_ref
    ctrl.U_ref[1] .= U_ref
    ctrl.K[1] .= K

end

function getcontrol(ctrl::TVLQRController, x, k; input_noise_type=nothing, input_noise_factor=0.0, dt=0.01)

    # calculate control input using ricatti
    dx = ctrl.state_error(x, ctrl.X_ref[1][k])
    u = ctrl.U_ref[1][k] - ctrl.K[1][k]*dx

    # determine noise
    w_u = 0
    
    if !isnothing(input_noise_type)
        w_u = input_noise_type == :u ? input_noise_factor.*u.*randn(length(u)) : (input_noise_factor/sqrt(dt)) .* randn(length(u))
    end

    u += w_u

    return u

end

function finite_horizon_ricatti(A_ref::AbstractArray{<:AbstractArray},B_ref::AbstractArray{<:AbstractArray},Q,Qf,R,N; verbose=false)

    # perform ricatti recursion    
    P = [zeros(size(A_ref[1], 1),size(A_ref[1], 1)) for i = 1:N]
    K = [zeros(size(B_ref[1], 2),size(A_ref[1], 1)) for i = 1:N-1]
    P[N] = Matrix(copy(Qf))

    for k = N-1:(-1):1

        A = A_ref[k]
        B = B_ref[k]

        K[k] = (R + B'*P[k+1]*B)\(B'*P[k+1]*A)
        P[k] = Q + A'*P[k+1]*(A - B*K[k])

    end
    
    return K,P

end