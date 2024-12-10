import Pkg; Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
using FileIO
using LinearPixelsToTorques
using LinearAlgebra
using SparseArrays
using StaticArrays
using JLD2
using Plots
using Random
using Statistics
using Distributions
using StatsBase
using RobotZoo
using MeshCat
using Test
using ImageShow
using ControlSystems
using IterTools
using Convex
using COSMO
using CairoMakie
import RobotDynamics as RD

if !isdir(LinearPixelsToTorques.DATA_DIR)
    mkdir(LinearPixelsToTorques.DATA_DIR)
end;
if !isdir(LinearPixelsToTorques.VIS_DIR)
    mkdir(LinearPixelsToTorques.VIS_DIR)
end;

####################################
## simulation parameters
####################################

sparse_data = true
stack_u = false
ones_and_zeros = false
learn_new_model = false

####################################
## define time properties
####################################

f = 60 # simulation frequency

dt = 1/f
tf = 2.5
N = Int(tf/dt)+1

####################################
## define training properties
####################################

num_train = 50
num_cross_validate = 15

# optimizer properties
max_iter = 10000

# image resolution (res x res)
resx = 160
resy = 125

####################################
## make cartpole model
####################################

model_nominal = RExCartpole()
model_real = RExCartpole(; mc=1.05 .* model_nominal.mc, mp=1.05 .* model_nominal.mp, l=model_nominal.l)

num_x = RD.state_dim(model_real)
num_u = RD.control_dim(model_real);
num_y = resx * resy

####################################
## make MeshCat visualization
####################################

println("Booting up Meshcat")
vis = Visualizer()
set_mesh!(vis, model_real)
visualize!(vis, model_real, SA[0., 0., 0., 0.])
render(vis)

####################################
## data collection setup
####################################

min_x = -model_nominal.l/3
max_x = model_nominal.l/3

min_ẋ = -0.05
max_ẋ = 0.05

min_θ = -deg2rad(15.0)
max_θ = deg2rad(15.0)

min_ω = -deg2rad(5)
max_ω = deg2rad(5)

x_min = [min_x, min_θ, min_ẋ, min_ω]
x_max = [max_x, max_θ, max_ẋ, max_ω]

# equilibrium point
x_eq = [0.0, 0.0, 0.0, 0.0]
u_eq = [0.0]

@test x_eq == dynamics_rk4(model_nominal, x_eq, u_eq, dt)

y_eq_pixel = get_pixel_state(
    model_real, x_eq, resx, resy;
    sparse_image=sparse_data,
    ones_and_zeros=ones_and_zeros
)
y_eq_pixel_greyscale = LinearPixelsToTorques.pixel_to_greyscale(y_eq_pixel, resx, resy)

#######################################################
## Create "teacher" controller and state estimator
#######################################################

A_true = dfdx(model_real, x_eq, u_eq, dt)
B_true = dfdu(model_real, x_eq, u_eq, dt)

A_nominal = LinearPixelsToTorques.dfdx(model_nominal, x_eq, u_eq, dt)
B_nominal = LinearPixelsToTorques.dfdu(model_nominal, x_eq, u_eq, dt)

# lqr controller
Q = Diagonal(vcat(50, 100, 1 .* ones(num_x-2)))
R = Diagonal([0.1])

teacher_controller = InfLQRController(model_nominal, Q, R, x_eq, u_eq, dt)

# Kalman filter
C = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]
R_cov = diagm([2, 2]) # measurement covariance
Q_cov = diagm([1, 1, 5, 5]) # process covariance

teacher_estimator = KalmanFilter(model_nominal, C, Q_cov, R_cov, x_eq, u_eq, dt)

#######################################################
## Execute teacher controller and collect data
#######################################################

Random.seed!(1)
training_data, test_data = generate_stabilizing_data(model_real, teacher_controller, teacher_estimator, x_min, x_max;
    test_window=1.0, num_train=200, num_test=100, tf=tf,
    dt=dt, resx=resx, resy=resy, visualize=false, sparse_image=sparse_data,
    process_noise_type=:dt, process_noise_factor=0.05,
    input_noise_type=nothing, input_noise_factor=0.0, measurement_noise_type=:encoder,
    measurement_noise_factor=[model_nominal.l, 2*pi]./600, ones_and_zeros=ones_and_zeros
);

jldsave(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_teacher_stabilizing_data.jld2");
    model_real, model_nominal, tf, dt, x_eq, u_eq, y_eq_pixel, training_data, test_data
)

#######################################################
## load training data
#######################################################

# load data
data = load(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_teacher_stabilizing_data.jld2"))
model_real = data["model_real"]
model_nominal = data["model_nominal"]
tf = data["tf"]
dt = data["dt"]
x_eq = data["x_eq"]
u_eq = data["u_eq"]
y_eq_pixel = data["y_eq_pixel"]
training_data = data["training_data"]
x0_test = data["test_data"]["x0_test"]

# extract training data from training set
training_ind = 1:150
Y_pixels_training = training_data["X_pixel_training_traj"][training_ind][1:num_train]
X_real_training = training_data["X_real_training_traj"][training_ind][1:num_train]
X_est_training = training_data["X_est_training_traj"][training_ind][1:num_train]
X_pred_training = training_data["X_pred_training_traj"][training_ind][1:num_train]
U_real_training = training_data["U_real_training_traj"][training_ind][1:num_train]
U_est_training = training_data["U_est_training_traj"][training_ind][1:num_train]
X0_train = training_data["x0_training"][training_ind][1:num_train]
T = training_data["T"]

# extract cross-validation data from training set
cross_val_ind = 151:200
Y_pixels_cross_val = training_data["X_pixel_training_traj"][cross_val_ind][1:num_cross_validate]
X_real_cross_val = training_data["X_real_training_traj"][cross_val_ind][1:num_cross_validate]
X_est_cross_val = training_data["X_est_training_traj"][cross_val_ind][1:num_cross_validate]
X_pred_cross_val = training_data["X_pred_training_traj"][cross_val_ind][1:num_cross_validate]
U_real_cross_val = training_data["U_real_training_traj"][cross_val_ind][1:num_cross_validate]
U_est_cross_val = training_data["U_est_training_traj"][cross_val_ind][1:num_cross_validate]
X0_cross_val = training_data["x0_training"][cross_val_ind][1:num_cross_validate]

#######################################################
## visualize a training trajectory 
#######################################################

traj_num = 1 # <<--- change this number around

X = [SVector{4}(x) for x in X_real_training[traj_num]]
visualize!(vis, model_real, tf, X)

################################################################
## visualize an image from a knot point of training trajectory
################################################################

knot_point = 1 # <<--- change this number around

y_pixel = Y_pixels_training[traj_num][knot_point]
y_greyscale_image = LinearPixelsToTorques.pixel_to_greyscale(y_pixel, resx, resy)
display(y_greyscale_image)

#######################################################
## plot the training trajectory 
#######################################################

x_real_hist = [x[1] for x in X]
θ_real_hist = [x[2] for x in X]
v_real_hist = [x[3] for x in X]
ω_real_hist = [x[4] for x in X]

x_est_hist = [x[1] for x in X_est_training[traj_num]]
θ_est_hist = [x[2] for x in X_est_training[traj_num]]
v_est_hist = [x[3] for x in X_est_training[traj_num]]
ω_est_hist = [x[4] for x in X_est_training[traj_num]]
u_real_hist = [u[1] for u in U_real_training[traj_num]]
u_est_hist = [u[1] for u in U_est_training[traj_num]]

fig1 = Plots.plot(T, [x_real_hist, θ_real_hist], label=["x (real)" "θ (real)"],
    xlabel="t", ylabel="state", title="Pendulum State", lw=2, legend=:outerright
)
Plots.plot!(fig1, T, [x_est_hist, θ_est_hist], label=["x (estimated)" "θ (estimated)"], lw=2)
Plots.plot!(fig1, T, [x_eq[1]*ones(length(T)), x_eq[2]*ones(length(T))],
    label=["x (goal)" "θ (goal)"], linestyle=:dash, lw=2
)

fig2 = Plots.plot(T, [v_real_hist, ω_real_hist], label=["v (real)" "ω (real)"],
    xlabel="t", ylabel="state", title="Pendulum State", lw=2, legend=:outerright
)
Plots.plot!(fig2, T, [v_est_hist, ω_est_hist], label=["v (estimated)" "ω (estimated)"], lw=2)
Plots.plot!(fig2, T, [x_eq[3]*ones(length(T)), x_eq[4]*ones(length(T))],
    label=["v (goal)" "ω (goal)"], linestyle=:dash, lw=2
)

fig3 = Plots.plot(T[1:end-1], u_real_hist, label="u (real)",
    xlabel="t", ylabel="control", title="Pendulum Control", lw=2
)
Plots.plot!(fig3, T[1:end-1], u_est_hist, label="u (estimated)", lw=2)

display(fig1)
display(fig2)
display(fig3);

#######################################################
## animate pixel image from training trajectory
#######################################################

filename = joinpath(LinearPixelsToTorques.VIS_DIR, "example_cartpole_stabilizing.gif")
animate_pixels(model_real, Y_pixels_training[traj_num], filename, resx, resy; framerate=f)

###########################################################
## get privileged info from teacher controller
###########################################################

K = teacher_controller.K

###################################
## learn (I-LC) and L matrices
###################################

println("Creating LLS problem...")

X_est_mat = concatenate_trajectories(X_est_training; start_ind=1, end_ind=1)
U_mat = concatenate_trajectories(U_est_training; start_ind=1)

num_data = size(X_est_mat, 2)

RHS = concatenate_trajectories(X_est_training; start_ind=2)
X_mat_1 = X_est_mat
X_mat_2 = concatenate_trajectories(Y_pixels_training; start_ind=2)

println("Solving LLS problem...")

ALK_hat = Variable(num_x, num_x)
L = Variable(num_x, num_y)
d = Variable(num_x)

λ = 1e-3 # <<--- L2 regularization parameter
problem = Convex.minimize(sumsquares(ALK_hat*X_mat_1 + L*X_mat_2 + d*ones(1, num_data) - RHS) + λ*norm(vec(L), 1))
solve!(problem, () -> COSMO.Optimizer(max_iter=max_iter))

println("Extracting matrices...")

ALK_hat = evaluate(ALK_hat)
L = evaluate(L)
d = evaluate(d)

println("Cross-validating...")

training_residual = norm(ALK_hat*X_mat_1 + L*X_mat_2 + d*ones(1, num_data) - RHS)

X_est_cross_val_mat = concatenate_trajectories(X_est_cross_val; start_ind=1, end_ind=1)
RHS_cross_val = concatenate_trajectories(X_est_cross_val; start_ind=2)
X_mat_2_cross_val = concatenate_trajectories(Y_pixels_cross_val; start_ind=2)
num_data_cross_val = size(X_est_cross_val_mat, 2)

test_residual = norm(ALK_hat*X_est_cross_val_mat + L*X_mat_2_cross_val + d*ones(1, num_data_cross_val) - RHS_cross_val)

@show(training_residual)
@show(test_residual)
@show(maximum(abs.(eigvals(ALK_hat))))

#######################################################
## save learned stuff
#######################################################

# save models
jldsave(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_stabilizing_example_data.jld2");
    model_real, model_nominal, K, L, ALK_hat, d, x_eq, u_eq, tf, dt 
)

#######################################################
## load models
#######################################################

data = load(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_stabilizing_example_data.jld2"));
model_real = data["model_real"];
model_nominal = data["model_nominal"];
K = data["K"];
L = data["L"];
ALK_hat = data["ALK_hat"];
d = data["d"];
x_eq = data["x_eq"];
u_eq = data["u_eq"];
tf = data["tf"];
dt = data["dt"];

############################################
## Create pixel-to-torques-policy instance
############################################

student_controller = deepcopy(teacher_controller)
student_estimator = LearnedFeedbackDynamicsLuenbergerObserver(ALK_hat, d, L);

println("student made!")

#######################################################
## execute pixel-to-torques policy on real physics
#######################################################

x0 = x0_test[1]
u0 = [0.0]

Random.seed!(3)
X_real_eval, X_est_eval, X_pred_eval, U_real_eval, U_est_eval, Y_pixel_eval, T_eval, successful_sim = simulatewithpixelcontrollerandestimator(
    model_real, student_controller, student_estimator, x0, 4*tf, dt; u0=u0,
    resx=resx, resy=resy, sparse_image=sparse_data,
    process_noise_type=:dt, process_noise_factor=0.05,
    input_noise_type=:dt, input_noise_factor=0.0, verbose=true,
    termination_criteria=(x) -> any(abs.(x[1:2]) .> [0.5, deg2rad(30)])
)

#######################################################
## visualize physical rollout results
#######################################################

X_eval = [SVector{4}(x) for x in X_real_eval]
visualize!(vis, model_real, 4*tf, X_eval)

#######################################################
## animate pixel image from trajectory
#######################################################

filename = joinpath(LinearPixelsToTorques.VIS_DIR, "learned_cartpole_stabilizing.gif")
animate_pixels(model_real, Y_pixel_eval, filename, resx, resy; framerate=f)

#######################################################
## plot physical rollout results
#######################################################

x_real_hist = [x[1] for x in X_real_eval]
θ_real_hist = [x[2] for x in X_real_eval]
v_real_hist = [x[3] for x in X_real_eval]
ω_real_hist = [x[4] for x in X_real_eval]

x_est_hist = [x[1] for x in X_est_eval]
θ_est_hist = [x[2] for x in X_est_eval]
v_est_hist = [x[3] for x in X_est_eval]
ω_est_hist = [x[4] for x in X_est_eval]

u_real_hist = [u[1] for u in U_real_eval]
u_est_hist = [u[1] for u in U_est_eval]

fig1 = Plots.plot(T_eval, [x_real_hist, θ_real_hist], label=["x (real)" "θ (real)"],
    xlabel="time (sec)", ylabel="Configuration", title="Pendulum State", lw=2, legend=:outerright,
    titlefontsize=10, xtickfontsize=10, ytickfontsize=10, legendfontsize=10, guidefontsize=10
)
Plots.plot!(fig1, T_eval, [x_est_hist, θ_est_hist], label=["x (estimated)" "θ (estimated)"], lw=2)
Plots.plot!(fig1, T_eval, [x_eq[1]*ones(length(T_eval)), x_eq[2]*ones(length(T_eval))],
    label=["x (goal)" "θ (goal)"], linestyle=:dash, lw=2
)

fig2 = Plots.plot(T_eval, [v_real_hist, ω_real_hist], label=["v (real)" "ω (real)"],
    xlabel="Time (sec)", ylabel="Velocity", lw=2, legend=:outerright,
    titlefontsize=10, xtickfontsize=10, ytickfontsize=10, legendfontsize=10, guidefontsize=10
)
Plots.plot!(fig2, T_eval, [v_est_hist, ω_est_hist], label=["v (estimated)" "ω (estimated)"], lw=2)
Plots.plot!(fig2, T_eval, [x_eq[3]*ones(length(T_eval)), x_eq[4]*ones(length(T_eval))],
    label=["v (goal)" "ω (goal)"], linestyle=:dash, lw=2
)

fig3 = Plots.plot(T_eval[1:end-1], u_real_hist, label="u (real)",
    xlabel="t", ylabel="control", title="Pendulum Control", lw=2
)
Plots.plot!(fig3, T_eval[1:end-1], u_est_hist, label="u (estimated)", lw=2)

display(fig1)
display(fig2)
display(fig3);

#######################################################
## post process result
#######################################################

max_distance_error = maximum(abs.(x_real_hist[end-(N÷5)+1:end]))
max_angle_error = rad2deg(maximum(abs.(θ_real_hist[end-(N÷5)+1:end])))

@show max_distance_error
@show max_angle_error

#######################################################
## L matrix visualization (Figure 6)
#######################################################

function get_rgba(img)
    return convert(Array{RGBA{Float64},2}, img)
end

p_L = L[1, :] ./ maximum(abs.(L[1, :]))
θ_L = L[2, :] ./ maximum(abs.(L[2, :]))
v_L = L[3, :] ./ maximum(abs.(L[3, :]))
ω_L = L[4, :] ./ maximum(abs.(L[4, :]))

x_coord = 1:resx
y_coord = resy:-1:1

p_L = reshape(p_L, resy, resx)'
v_L = reshape(v_L, resy, resx)'
θ_L = reshape(θ_L, resy, resx)'
ω_L = reshape(ω_L, resy, resx)'

y_eq_pixel = get_pixel_state(
    model_real, x_eq, resx, resy;
    sparse_image=true,
    ones_and_zeros=ones_and_zeros
)
y_eq_pixel_greyscale = LinearPixelsToTorques.pixel_to_greyscale(y_eq_pixel, resx, resy)

# plot and save position gain
fig, ax, hm = Makie.heatmap(x_coord, y_coord, p_L, colormap=:bwr,
    show_axis=false, colorrange=[-1, 1]; axis=(; aspect=DataAspect())
)
hidedecorations!(ax)
# Colorbar(fig[:, end+1], hm, ticks = -1:0.5:1,
#     ticklabelsize=40, ticklabelfont = "Times New Roman")
# rowsize!(fig.layout, 1, ax.scene.px_area[].widths[2])
display(fig)
save(joinpath(LinearPixelsToTorques.VIS_DIR, "figure_4a_no_cart.svg"), fig)

# plot and save angle gain
fig, ax, hm = Makie.heatmap(x_coord, y_coord, θ_L, colormap=:bwr,
    show_axis=false, colorrange=[-1, 1]; axis=(; aspect=DataAspect())
)
hidedecorations!(ax)
# Colorbar(fig[:, end+1], hm, ticks = -1:0.5:1,
#     ticklabelsize=40, ticklabelfont = "Times New Roman")
# rowsize!(fig.layout, 1, ax.scene.px_area[].widths[2])
display(fig)
save(joinpath(LinearPixelsToTorques.VIS_DIR, "figure_4b_no_cart.svg"), fig)

# plot and save velocity gain
fig, ax, hm = Makie.heatmap(x_coord, y_coord, v_L, colormap=:bwr,
    show_axis=false, colorrange=[-1, 1]; axis=(; aspect=DataAspect())
)
hidedecorations!(ax)
# Colorbar(fig[:, end+1], hm, ticks = -1:0.5:1,
#     ticklabelsize=40, ticklabelfont = "Times New Roman")
# rowsize!(fig.layout, 1, ax.scene.px_area[].widths[2])
display(fig)
save(joinpath(LinearPixelsToTorques.VIS_DIR, "figure_4c_no_cart.svg"), fig)

# plot and save angular velocity gain
fig, ax, hm = Makie.heatmap(x_coord, y_coord, ω_L, colormap=:bwr,
    show_axis=false, colorrange=[-1, 1]; axis=(; aspect=DataAspect())
)
hidedecorations!(ax)
# Colorbar(fig[:, end+1], hm, ticks = -1:0.5:1, tellheight=false,
#     ticklabelsize=40, ticklabelfont = "Times New Roman")
# rowsize!(fig.layout, 1, ax.scene.px_area[].widths[2])
display(fig)
save(joinpath(LinearPixelsToTorques.VIS_DIR, "figure_4d_no_cart.svg"), fig)