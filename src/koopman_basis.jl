function hermite(x::AbstractVector{T}; order::Vector{Int64} = [0, 0],
    scale = false, domain = [-ones(length(x)), ones(length(x))]
) where T

    if order == [0, 0]
        return T[]
    end

    if scale
        domain_mean = (domain[1] + domain[2]) ./ 2
        domain_mag = domain[2] - domain_mean

        x = (x .- domain_mean) ./ domain_mag
    end

    T0 = ones(T, length(x))
    T1 = 2 .* x

    hermite_poly = [T0]
    push!(hermite_poly, T1)

    for p in 2:order[2]

        next_T = (2 .* x .* hermite_poly[p]) - 2 .* p .* hermite_poly[p-1]
        push!(hermite_poly, next_T)

    end

    hermite_poly = reduce(vcat, hermite_poly[order[1]+1:order[2]+1])

    return hermite_poly
    
end

function chebyshev(x::AbstractVector{T}; order::Vector{Int64} = [0, 0],
    scale = false, domain = [-ones(length(x)), ones(length(x))]
) where T

    if order == [0, 0]
        return T[]
    end

    if scale
        domain_mean = (domain[1] + domain[2]) ./ 2
        domain_mag = domain[2] - domain_mean

        x = (x .- domain_mean) ./ domain_mag
    end

    T0 = ones(T,length(x))
    T1 = x

    chebyshev_poly = [T0]
    push!(chebyshev_poly, T1)

    for p in 2:order[2]

        next_T = (2 .* x .* chebyshev_poly[p]) - chebyshev_poly[p-1]
        push!(chebyshev_poly, next_T)

    end

    chebyshev_poly = reduce(vcat, chebyshev_poly[order[1]+1:order[2]+1])

    return chebyshev_poly
    
end

function monomial(x::AbstractVector{T}; order::Vector{Int64} = [0, 0],
    scale = false, domain=[]) where T

    if order == [0, 0]
        return T[]
    end

    T0 = ones(T,length(x))

    monomials = [T0]
    push!(monomials, x)

    row_start_ind = ones(T,length(x))

    for p in 2:order[2]

        prev_row_start_ind = row_start_ind

        mono_combinations = x * monomials[p]'
        mono_permutations = mono_combinations[1, :]
        row_start_ind = [size(mono_combinations)[2]]

        for i in 2:size(mono_combinations)[1]

            row_start_ind = vcat(row_start_ind, Int(row_start_ind[i-1] - prev_row_start_ind[i-1]))
            current_row_permutations = mono_combinations[i, end-row_start_ind[i]+1:end]
            mono_permutations = vcat(mono_permutations, current_row_permutations)

        end

        push!(monomials, mono_permutations)
    
    end

    monomials = reduce(vcat, monomials[order[1]+1:order[2]+1])

    return monomials
        
end

function fourier(x::AbstractVector{T}; order::Vector{Int64} = [0, 0],
    scale = false, domain = [zeros(length(x)), 2*pi.*ones(length(x))]
) where T

    if order == [0, 0]
        return T[]
    end

    if scale
        P = domain[2] - domain[1]
    else
        P = 2*pi
    end

    T0 = ones(T,length(x))

    fourier_poly = [T0]

    for p in 1:order[2]

        next_T = vcat(sin.((2*pi ./ P .* p) .* x), cos.((2*pi ./ P .* p) .* x))
        push!(fourier_poly, next_T)

    end

    fourier_poly = reduce(vcat, fourier_poly[order[1]+1:order[2]+1])

    return fourier_poly
        
end

state(xk::AbstractVector; scale=false, domain=[], order=[0]) = xk
sine(xk::AbstractVector; scale=false, domain=[], order=[1]) = sin.(order[1]*xk)
cosine(xk::AbstractVector; scale=false, domain=[], order=[1]) = cos.(order[1]*xk)

state_transform(z::AbstractVector, g) = g * z

function koopman_transform(x::AbstractVector{<:AbstractFloat},
    function_list::Vector{String},
    states_list::Vector{Vector{Bool}}, 
    order_list::Vector{Vector{Int64}};
    scale_list = zeros(Bool, length(function_list)),
    domain = [-ones(length(x)), ones(length(x))]
)

    num_func = length(function_list)
    z = Vector{Float64}()

    for i in 1:num_func

        func = function_list[i]
        order = order_list[i]
        scale = scale_list[i]
        applied_states = states_list[i]

        applied_x = x[applied_states]
        applied_domain = [domain[1][applied_states], domain[2][applied_states]]

        a = eval(Symbol(func))
        
        if order == [0, 0] || order == [0]
            func_eval = a(applied_x; scale=scale, domain=applied_domain)
        else
            func_eval = a(applied_x; order=order, scale=scale, domain=applied_domain)
        end
                
        append!(z, func_eval)

    end

    return z

end

function koopman_transform(X::Vector{<:AbstractVector},
    function_list::Vector{String},
    states_list::Vector{Vector{Bool}}, 
    order_list::Vector{Vector{Int64}};
    scale_list = zeros(Bool, length(function_list)),
    domain = [-ones(length(X[1])), ones(length(X[1]))]
)

    Z = [koopman_transform(x, function_list, states_list, order_list; scale_list=scale_list, domain=domain) for x in X]

    return Z

end

function bilinear_koopman_transform(X::Vector{<:AbstractVector},
    U::Vector{<:AbstractVector},
    function_list::Vector{String},
    states_list::Vector{Vector{Bool}}, 
    order_list::Vector{Vector{Int64}};
    scale_list = zeros(Bool, length(function_list)),
    domain = [-ones(length(X[1])), ones(length(X[1]))]
)

    Z = [koopman_transform(x, function_list, states_list, order_list; scale_list=scale_list, domain=domain) for x in X]
    Zu = map(zip(CartesianIndices(U), U)) do (cind,u)
        vcat(Z[cind], u, vec(Z[cind]*u'))
    end

    return Zu

end