struct RExCartpole <: RD.AbstractModel
    mc::Float64
    mp::Float64
    l::Float64
    g::Float64
end

RExCartpole(; mc=0.4, mp=0.307, l=0.352, g=9.81) = RExCartpole(mc, mp, l, g)

function RD.dynamics(model::RExCartpole, x, u)
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    r = x[1] # cart position
    θ = x[2] # pole angle
    rd = x[3] # change in cart position
    θd = x[4] # change in pole angle
    F = u[1] # force applied to cart
    
    θdd = (g*sin(θ) + cos(θ) * ((-F - mp*l*(θd^2) * sin(θ))/(mc + mp))) / (l*(4/3 - (mp*(cos(θ)^2))/(mc + mp)))
    rdd = (F + mp*l*((θd^2)*sin(θ) - θdd*cos(θ))) / (mc + mp)
  
    return [rd; θd; rdd; θdd]
    
end

function set_mesh!(vis::Visualizer, model::RExCartpole; color=nothing, color2=nothing)
    cartpole_plot = RobotZoo.Cartpole(; mc=model.mc, mp=model.mp, l=model.l, g=model.g)
    set_mesh!(vis, cartpole_plot; color=color, color2=color2)

    return nothing

end

function visualize!(vis::Visualizer, model::RExCartpole, x::StaticArray{Tuple{N}, T, 1} where {N, T})

    cartpole_plot = RobotZoo.Cartpole(; mc=model.mc, mp=model.mp, l=model.l, g=model.g)
    x_plot = [-x[1], x[2]+pi, x[3], x[4]]

    visualize!(vis, cartpole_plot, SVector{4,Float64}(x_plot))

    return nothing

end

RD.state_dim(::RExCartpole) = 4
RD.control_dim(::RExCartpole) = 1