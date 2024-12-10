import Pkg;
Pkg.activate(joinpath(@__DIR__, ".."));
Pkg.instantiate()

using LinearAlgebra
using SparseArrays
using StaticArrays
using ForwardDiff
using NPZ
using JLD2
using Convex
using COSMO
import RobotDynamics as RD
using LinearPixelsToTorques
using MeshCat
using Plots
using ImageShow
using Random
using Statistics
using Distributions
using ProgressMeter
using CairoMakie

const MOI = Convex.MOI

if !isdir(LinearPixelsToTorques.DATA_DIR)
    mkdir(LinearPixelsToTorques.DATA_DIR)
end;
if !isdir(LinearPixelsToTorques.VIS_DIR)
    mkdir(LinearPixelsToTorques.VIS_DIR)
end;

####################################
## import data
####################################

data_file_name = "hardware_cartpole_stabilizing_training_data.npz";
full_data_file_path = joinpath(LinearPixelsToTorques.DATA_DIR, data_file_name);
data = npzread(full_data_file_path);

start_ind = 3 .+ data["traj_snippet_start_indices"];
end_ind = data["traj_snippet_end_indices"];

resy, resx = size(data["frames"][:, :, 1])

X_est_mat = data["est_states_encoder"];
U_mat = reshape(data["controls_encoder"], 1, length(data["controls_encoder"]));

Y_pixels_list = [vec(convert(Matrix{Float64}, data["frames"][:, :, i])) for i in 1:size(data["frames"], 3)];
Y_pixels_mat = sparse(reduce(hcat, Y_pixels_list))

X_est_data = [[X_est_mat[:, start_ind[i]:end_ind[i]][:, j] for j in 1:size(X_est_mat[:, start_ind[i]:end_ind[i]], 2)] for i in 1:length(start_ind)]
U_est_data = [[U_mat[:, start_ind[i]:end_ind[i]-1][:, j] for j in 1:size(U_mat[:, start_ind[i]:end_ind[i]-1], 2)] for i in 1:length(start_ind)]
Y_pixels_data = [[Y_pixels_mat[:, start_ind[i]:end_ind[i]][:, j] for j in 1:size(Y_pixels_mat[:, start_ind[i]:end_ind[i]], 2)] for i in 1:length(start_ind)]
Y_all_pixels_traj = [Y_pixels_mat[:, i] for i in 1:size(Y_pixels_mat, 2)]

#######################################################
## save data in julia format
#######################################################

# save models
jldsave(joinpath(LinearPixelsToTorques.DATA_DIR, "hardware_cartpole_stabilizing_training_data.jld2");
    resy, resx, X_est_data, U_est_data, Y_pixels_data, Y_all_pixels_traj
)

#######################################################
## learning setup
#######################################################

model_nominal = RExCartpole()

# optimizer properties
max_iter = 10000
num_train = 20
num_cross_validate = 4

num_x = RD.state_dim(model_nominal)
num_u = RD.control_dim(model_nominal);

K = 0.2 .* [-3.29489, -19.36664, -4.46195, -2.79850/1.5]'

#######################################################
## load models
#######################################################

data = load(joinpath(LinearPixelsToTorques.DATA_DIR, "hardware_cartpole_stabilizing_training_data.jld2"));
resy = data["resy"]
resx = data["resx"]
num_y = resx * resy

# extract training data
Random.seed!(1)
training_ind = sample(1:length(data["X_est_data"]), num_train, replace=false, ordered=true)

X_est_training = data["X_est_data"][training_ind]
U_est_training = data["U_est_data"][training_ind]
Y_pixels_training = data["Y_pixels_data"][training_ind]
Y_all_pixels_traj = data["Y_all_pixels_traj"]
num_train = length(X_est_training)

cross_val_ind = sample(setdiff(1:length(data["X_est_data"]), training_ind), num_cross_validate, replace=false, ordered=true)

X_est_cross_val = data["X_est_data"][cross_val_ind]
U_est_cross_val = data["U_est_data"][cross_val_ind]
Y_pixels_cross_val = data["Y_pixels_data"][cross_val_ind]

####################################
## make MeshCat visualization
####################################

println("Booting up Meshcat")
vis = Visualizer()
set_mesh!(vis, model_nominal)
visualize!(vis, model_nominal, SA[0., 0., 0., 0.])
render(vis)

#######################################################
## visualize trajectory
#######################################################

traj_num = 1

X = [SVector{4}(x) for x in X_est_training[traj_num]]
U = U_est_training[traj_num]
Y = Y_pixels_training[traj_num]

visualize!(vis, model_nominal, length(X)÷60, X)

#######################################################
## visualize an image from a knot point of trajectory
#######################################################

knot_point = 40 # <<--- change this number around

y_pixel = Y_pixels_training[traj_num][knot_point]
y_greyscale_image = LinearPixelsToTorques.pixel_to_greyscale(y_pixel, resx, resy)
display(y_greyscale_image)

#######################################################
## animate pixel image from trajectory
#######################################################

all_traj_Y_gray = [LinearPixelsToTorques.pixel_to_greyscale(y, resx, resy) for y in Y_all_pixels_traj[210:210 + 60*30]]

filename = joinpath(LinearPixelsToTorques.VIS_DIR, "hardware_cartpole_training_demonstrations.mp4")
scene = construct_scene(resx, resy)
frames = 1:length(all_traj_Y_gray)

p = Progress(length(frames), 1, "Creating animation...")

record(scene, filename, frames, framerate=60) do i

    y_i = all_traj_Y_gray[i]
    y_i[y_i .< 0.0] .= 0.0
    y_i[y_i .> 1.0] .= 1.0

    Makie.image!(scene, rotr90(y_i))

    next!(p)

end

#######################################################
## plot the trajectory
#######################################################

x_est_hist = [x[1] for x in X]
θ_est_hist = [x[2] for x in X]
v_est_hist = [x[3] for x in X]
ω_est_hist = [x[4] for x in X]
u_est_hist = [u[1] for u in U]
T_plot = 0:1/60:(length(X)-1)/60

fig1 = Plots.plot(T_plot, [x_est_hist, θ_est_hist], label=["x (est)" "θ (est)"],
    xlabel="t", ylabel="state", title="Pendulum State", lw=2, legend=:outerright
)

fig2 = Plots.plot(T_plot, [v_est_hist, ω_est_hist], label=["v (est)" "ω (est)"],
    xlabel="t", ylabel="state", title="Pendulum State", lw=2, legend=:outerright
)

fig3 = Plots.plot(T_plot[1:end-1], u_est_hist, label="u",
    xlabel="t", ylabel="control", title="Pendulum Control", lw=2
)

display(fig1)
display(fig2)
display(fig3);

#######################################################
## plot position and angle for all trajectories
#######################################################

for X in data["X_est_data"]
    x_est_hist = [x[1] for x in X]
    θ_est_hist = [x[2] for x in X]
    v_est_hist = [x[3] for x in X]

    ω_est_hist = [x[4] for x in X]
    T_plot = 0:1/60:(length(X)-1)/60

    fig1 = Plots.plot(T_plot, [x_est_hist, θ_est_hist], label=["x (est)" "θ (est)"],
        xlabel="t", ylabel="state", title="Pendulum State", lw=2, legend=:outerright
    )

    display(fig1)
end

#######################################################
## plot control for all trajectories
#######################################################

for U in data["U_est_data"]
    u_est_hist = [u[1] for u in U]
    T_plot = 0:1/60:(length(U)-1)/60

    fig1 = Plots.plot(T_plot, u_est_hist, label="u",
        xlabel="t", ylabel="control", title="Pendulum Control", lw=2
    )

    display(fig1)
end

##################################################################
## learn LO matrices
##################################################################

println("Creating LLS problem...")

X_est_mat = concatenate_trajectories(X_est_training; start_ind=1, end_ind=1)

num_data = size(X_est_mat, 2)

RHS = concatenate_trajectories(X_est_training; start_ind=2)
X_mat = X_est_mat
X_mat_3 = concatenate_trajectories(Y_pixels_training; start_ind=2)

println("Solving LLS problem...")

ABK_hat = Variable(num_x, num_x)
L = Variable(num_x, num_y)
d = Variable(num_x)

λ = 2e-2
problem = minimize(sumsquares(ABK_hat*X_mat + L*X_mat_3 + d*ones(1, num_data) - RHS) + λ*norm(vec(L), 1),
    isposdef([I(num_x) ABK_hat; ABK_hat' I(num_x)]),
)
@time solve!(problem, () -> COSMO.Optimizer(max_iter=max_iter))

println("Extracting matrices...")

ABK_hat = evaluate(ABK_hat)
L = evaluate(L)
d = evaluate(d)

println("Cross-validating...")
training_residual = norm(ABK_hat*X_est_mat + L*X_mat_3 + d*ones(1, num_data) - RHS)

X_est_cross_val_mat = concatenate_trajectories(X_est_cross_val; start_ind=1, end_ind=1)
RHS_cross_val = concatenate_trajectories(X_est_cross_val; start_ind=2)
X_mat_3_cross_val = concatenate_trajectories(Y_pixels_cross_val; start_ind=2)
num_data_cross_val = size(X_est_cross_val_mat, 2)

test_residual = norm(ABK_hat*X_est_cross_val_mat + L*X_mat_3_cross_val + d*ones(1, num_data_cross_val) - RHS_cross_val)

@show(training_residual)
@show(test_residual)
@show(maximum(abs.(eigvals(ABK_hat))))

#######################################################
## save learned stuff
#######################################################

# save models
data_write_file = "learned_hardware_lo_matrices_2e-2_reg_assume_feedback_20_traj.npy"

npzwrite(joinpath(LinearPixelsToTorques.DATA_DIR, data_write_file),
    Dict("ABK_hat" => ABK_hat, "L" => L, "d2" => d)
)