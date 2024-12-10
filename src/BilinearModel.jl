
struct BilinearModel <: RD.AbstractModel
    A::Union{SparseMatrixCSC{Float64, Int64}, Matrix{Float64}}
    B::Union{SparseMatrixCSC{Float64, Int64}, Matrix{Float64}}
    C::Union{Vector{SparseMatrixCSC{Float64, Int64}}, Vector{Matrix{Float64}}}
    d::Union{SparseVector{Float64, Int64}, Vector{Float64}}
    dt::Float64
    name::String

    function BilinearModel(A::AbstractMatrix, B::AbstractMatrix, C::Vector{<:AbstractMatrix}, 
                       d::AbstractVector, dt::AbstractFloat, name::AbstractString)
        p, n = size(A);
        m = length(C);
        p == size(B,1) || throw(DimensionMismatch("B should have the same number of rows as A."));
        if size(B,2) == 0
            B = zeros(p,m)
        end
        print("before C");
        all(c->size(c) == (p,n), C) || throw(DimensionMismatch("All C matrices should be the same size as A."));
        print("done");
        new(A, B, C, d, dt, name);
    end
end

function BilinearModel(A::AbstractMatrix, C::Vector{<:AbstractMatrix},
    d::AbstractVector, dt::AbstractFloat, name::AbstractString)
    n = size(A,2);
    m = size(C,1);
    B = zeros(n,m);
    BilinearModel(A,B,C,d,dt,name);
end

RD.state_dim(model::BilinearModel) = size(model.A,2)
RD.control_dim(model::BilinearModel) = size(model.B,2)

Base.copy(model::BilinearModel) = BilinearModel(copy(model.A), copy(model.B), copy(model.C), copy(model.d), model.dt, model.name)

function discrete_dynamics(model::BilinearModel, x, u, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."
    return model.A*x .+ model.B*u .+ model.d .+ sum(model.C[i]*x .* u[i] for i in eachindex(u))
end

function discrete_jacobian(model::BilinearModel, x, u, h)
    @assert h ≈ model.dt "Timestep must be $(model.dt)."

    m = size(model.B,2)

    dfdx = copy(model.A)
    dfdu = copy(model.B)

    for i = 1:m
       dfdx .+= model.C[i] .* u[i]
       mul!(dfdu, model.C[i], x, true, true)
    end

    return dfdx, dfdu

end