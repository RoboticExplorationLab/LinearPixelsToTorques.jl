using LinearAlgebra
using SparseArrays

function concatenate_trajectories(X_list::VecOrMat{<:VecOrMat{<:AbstractVector}}, x_eq::AbstractVector;
    start_ind=1, end_ind=0
)

    x_eq_list = [x_eq for i in 1:length(X_list)]
    X_mat_list = concatenate_trajectory.(X_list, x_eq_list; start_ind=start_ind, end_ind=end_ind)

    X_mat = reduce(hcat, X_mat_list)

    return X_mat

end

function concatenate_trajectory(X::VecOrMat{<:AbstractVector}, x_eq::AbstractVector;
    start_ind=1, end_ind=0
)

    X_mat = reduce(hcat, X[start_ind:end-end_ind]) .- kron(ones(1, length(X)-end_ind-start_ind+1), x_eq)

    return X_mat
end

function concatenate_trajectories(X_list::VecOrMat{<:VecOrMat{<:AbstractVector}}; start_ind=1,
    end_ind=0
)

    X_mat_list = concatenate_trajectory.(X_list; start_ind=start_ind, end_ind=end_ind)

    X_mat = reduce(hcat, X_mat_list)

    return X_mat

end

function concatenate_trajectory(X::VecOrMat{<:AbstractVector}; start_ind=1,
    end_ind=0
)

    X_mat = reduce(hcat, X[start_ind:end-end_ind])

    return X_mat
end