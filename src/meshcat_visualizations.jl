using GeometryBasics
using Rotations
using CoordinateTransformations
using StaticArrays
using FileIO
using Colors
using RobotDynamics
using MeshCat

function defcolor(c1, c2, c1def, c2def)
    if !isnothing(c1) && isnothing(c2)
        c2 = c1
    else
        c1 = isnothing(c1) ? c1def : c1
        c2 = isnothing(c2) ? c2def : c2
    end
    c1,c2
end

function set_mesh!(vis, model::RobotDynamics.AbstractModel; kwargs...)
    _set_mesh!(vis["robot"], model; kwargs...)
end

RobotDynamics.RBState(model::RobotDynamics.AbstractModel, x) = 
    RBState(position(model, x), orientation(model, x), zeros(3), zeros(3))

function _set_mesh!(vis, model::RobotZoo.DoubleIntegrator{<:Any,2}; 
        color=colorant"green", radius=0.1, height=0.05)
    radius = Float32(radius) 
    body = Cylinder(Point3f0(0,0,0), Point3f0(0,0,height), radius)
    setobject!(vis["geom"]["body"], body, MeshPhongMaterial(color=color))
end

# Cartpole
function _set_mesh!(vis, model::RobotZoo.Cartpole; 
        color=nothing, color2=nothing)
    dim = Vec(0.1, 0.3, 0.1)
    rod = Cylinder(Point3f0(0,-10,0), Point3f0(0,10,0), 0.01f0)
    cart = Rect3D(-dim/2, dim)
    hinge = Cylinder(Point3f0(-dim[1]/2,0,dim[3]/2), Point3f0(dim[1],0,dim[3]/2), 0.03f0)
    c1,c2 = defcolor(color,color2, colorant"blue", colorant"red")

    pole = Cylinder(Point3f0(0,0,0),Point3f0(0,0,model.l),0.01f0)
    mass = HyperSphere(Point3f0(0,0,model.l), 0.05f0)
    setobject!(vis["rod"], rod, MeshPhongMaterial(color=colorant"grey"))
    setobject!(vis["cart","box"],   cart, MeshPhongMaterial(color=isnothing(color) ? colorant"green" : color))
    setobject!(vis["cart","hinge"], hinge, MeshPhongMaterial(color=colorant"black"))
    setobject!(vis["cart","pole","geom","cyl"], pole, MeshPhongMaterial(color=c1))
    setobject!(vis["cart","pole","geom","mass"], mass, MeshPhongMaterial(color=c2))
    settransform!(vis["cart","pole"], Translation(0.75*dim[1],0,dim[3]/2))
end

function visualize!(vis::Visualizer, model::RobotZoo.Cartpole, x::StaticVector)
    y = x[1]
    θ = x[2]
    q = expm((pi-θ) * @SVector [1,0,0])
    settransform!(vis["robot","cart"], Translation(0,-y,0))
    settransform!(vis["robot","cart","pole","geom"], LinearMap(UnitQuaternion(q)))
end

## Visualize trajectory
function visualize!(vis::Visualizer, model, tf, X; color=nothing, color2=nothing)
    N = length(X)
    fps = Int(floor((N-1)/tf))
    anim = MeshCat.Animation(fps)
    set_mesh!(vis, model, color=color, color2=color2)
    for k = 1:N
        atframe(anim, k) do
            visualize!(vis, model, X[k]) 
        end
    end
    setanimation!(vis, anim)
end