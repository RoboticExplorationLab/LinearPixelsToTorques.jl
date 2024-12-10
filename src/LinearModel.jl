
struct LinearModel <: RD.AbstractModel
    A::Union{SparseMatrixCSC{Float64}, Matrix{Float64}}
    B::Union{SparseMatrixCSC{Float64}, Matrix{Float64}}
    d::Union{SparseVector{Float64}, Vector{Float64}}
    dt::Float64
    name::String

    function LinearModel(A::AbstractMatrix, B::AbstractMatrix, 
                       d::AbstractVector, dt::AbstractFloat, name::AbstractString)

        p,_ = size(A)
        p == size(B,1) || throw(DimensionMismatch("B should have the same number of rows as A."))
        new(A,B,d,dt,name)
    end

end

RD.state_dim(model::LinearModel) = size(model.A,2)
RD.control_dim(model::LinearModel) = size(model.B,2)

Base.copy(model::LinearModel) = LinearModel(copy(model.A), copy(model.B), copy(model.d), model.dt, model.name)

function discrete_dynamics(model::LinearModel, x, u, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."
    return model.A*x .+ model.B*u + model.d
end

function discrete_jacobian(model::LinearModel, x, u, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."

    dfdx = copy(model.A)
    dfdu = copy(model.B)

    return dfdx, dfdu

end