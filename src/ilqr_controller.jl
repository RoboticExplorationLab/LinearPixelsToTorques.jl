##############################
## iLQR
##############################

struct iLQRController <: AbstractController
    model::RobotDynamics.AbstractModel
    Q::Matrix{Float64}
    Qf::Matrix{Float64}
    R::Matrix{Float64}
    dt::Float64
    tf::Float64
    N::Vector{Int64}
end

function iLQRController(
    model::RobotDynamics.AbstractModel, 
    Q, Qf, R, dt, tf
)

    N = tf÷dt + 1

    iLQRController(model, Matrix(Q), Matrix(Qf), Matrix(R), dt, tf, [N])

end

function generate_trajectory(method::iLQRController, x0, u0, xgoal, ugoal, dt, tf; max_iter=3000)

    method.N[1] = Int(tf/dt) + 1

    X_traj, U_traj, T = ilqr_solve(method, x0, u0, xgoal; max_iter=max_iter)

    return X_traj, U_traj, T

end


function ilqr_solve(controller::iLQRController, x0, u0, xgoal; max_iter=3000)

    model = controller.model
    Q = controller.Q
    Qf = controller.Qf
    R = controller.R
    dt = controller.dt
    N = controller.N[1]
    
    @assert length(x0) == RobotDynamics.state_dim(model)
    @assert length(u0) == RobotDynamics.control_dim(model)

    Nx = RobotDynamics.state_dim(model)
    Nu = RobotDynamics.control_dim(model)
    
    T = Array(range(0, dt*(N-1), step=dt))

    X_traj = [x0 for _ in 1:N]
    U_traj = [u0 for _ in 1:(N-1)]

    # Initial rollout 
    for k = 1:N-1
        X_traj[k+1] = dynamics_rk4(model, X_traj[k], U_traj[k], dt)
    end

    J = quadratic_cost(xgoal, X_traj, U_traj, N, Q, R, Qf)

    p = [ones(Nx) for _ in 1:N]
    P = [zeros(Nx,Nx) for _ in 1:N]
    d = [ones(Nu) for _ in 1:(N-1)]
    K = [zeros(Nu, Nx) for _ in 1:N-1]

    Xn = [zeros(Nx) for _ in 1:N]
    Un = [zeros(Nu) for _ in 1:N-1]

    iter = 0
    while maximum(norm.(d, 1)) > 1e-3 && iter < max_iter
        iter += 1    
        
        #Backward Pass
        ΔJ = backward_pass!(model, p, P, d, K, Q, R, Qf, X_traj, U_traj, xgoal, N, dt)

        #Forward rollout with line search
        Xn[1] = X_traj[1]
        α = 1.0

        for k = 1:(N-1)
            Un[k] = U_traj[k] - α*d[k] - K[k]*(Xn[k]-X_traj[k])
            Xn[k+1] = dynamics_rk4(model, Xn[k], Un[k], dt)
        end

        Jn = quadratic_cost(xgoal, Xn, Un, N, Q, R, Qf)
        
        while isnan(Jn) || Jn > (J - 1e-2*α*ΔJ)
            α = 0.5*α
            for k = 1:(N-1)
                Un[k] = U_traj[k] - α*d[k] - K[k]*(Xn[k]-X_traj[k])
                Xn[k+1] = dynamics_rk4(model, Xn[k], Un[k], dt)
            end
            Jn = quadratic_cost(xgoal, Xn, Un, N, Q, R, Qf)
        end
        
        J = Jn
        X_traj .= Xn
        U_traj .= Un
    end
 
    return X_traj, U_traj, T
end

function backward_pass!(model,p, P, d, K, Q, R, Qf, X_traj, U_traj, xgoal, Nt, dt; gauss_newton=true)
    
    ΔJ = 0.0
    p[Nt] .= Qf*(X_traj[Nt]-xgoal)
    P[Nt] .= Qf
    
    for k = (Nt-1):-1:1
        #Calculate derivatives
        q = Q*(X_traj[k]-xgoal)
        r = R*U_traj[k]

        A = dfdx(model, X_traj[k], U_traj[k], dt) 
        B = dfdu(model, X_traj[k], U_traj[k], dt)
        
        gx = q + A'*p[k+1]
        gu = r + B'*p[k+1]

        
        #iLQR (Gauss-Newton) version
        Gxx = Q + A'*P[k+1]*A
        Guu = R + B'*P[k+1]*B
        Gxu = A'*P[k+1]*B
        Gux = B'*P[k+1]*A

        
        if !gauss_newton
            #DDP (full Newton) version
            Ax = dAdx(model, X_traj[k], U_traj[k], dt)
            Bx = dBdx(model, X_traj[k], U_traj[k], dt)
            Au = dAdu(model, X_traj[k], U_traj[k], dt)
            Bu = dBdu(model, X_traj[k], U_traj[k], dt)
            Gxx += kron(p[k+1]',I(Nx))*comm(Nx,Nx)*Ax
            Guu += kron(p[k+1]',I(Nu))*comm(Nx,Nu)*Bu[1]
            Gxu += kron(p[k+1]',I(Nx))*comm(Nx,Nx)*Au
            Gux += kron(p[k+1]',I(Nu))*comm(Nx,Nu)*Bx
        end

        β = 0.1
        while !isposdef(Symmetric([Gxx Gxu; Gux Guu]))
            Gxx += A'*β*I*A
            Guu += B'*β*I*B
            Gxu += A'*β*I*B
            Gux += B'*β*I*A
            β = 2*β
            display("regularizing G")
        end
        
        d[k] .= Guu\gu
        K[k] .= Guu\Gux
    
        p[k] .= gx - K[k]'*gu + K[k]'*Guu*d[k] - Gxu*d[k]
        P[k] .= Gxx + K[k]'*Guu*K[k] - Gxu*K[k] - K[k]'*Gux
    
        ΔJ += gu'*d[k]
    end
    
    return ΔJ
end

##############################
## Costs
##############################

function quadratic_stage_cost(x, u, xgoal, Q, R)
    return 0.5*((x-xgoal)'*Q*(x-xgoal)) + 0.5*u'*R*u
end

function quadratic_terminal_cost(x, xgoal, Qf)
    return 0.5*((x-xgoal)'*Qf*(x-xgoal))
end

function quadratic_cost(xgoal, X_traj, U_traj, Nt, Q, R, Qf)
    J = 0.0
    for k = 1:Nt-1
        J += quadratic_stage_cost(X_traj[k], U_traj[k], xgoal, Q, R)
    end
    J += quadratic_terminal_cost(X_traj[Nt], xgoal, Qf)
    return J
end