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
using RobotZoo
using Distributions
using MeshCat
using CairoMakie
using Test
using ImageShow
using ControlSystems
using IterTools
using Convex
using COSMO
import RobotDynamics as RD

if !isdir(LinearPixelsToTorques.DATA_DIR)
    mkdir(LinearPixelsToTorques.DATA_DIR)
end;
if !isdir(LinearPixelsToTorques.VIS_DIR)
    mkdir(LinearPixelsToTorques.VIS_DIR)
end;

####################################
## Simulation Parameters
####################################

sparse_data = true
stack_u = false
ones_and_zeros = false
learn_new_model = false

####################################
## define time properties
####################################

f = 30 # simulation frequency

dt = 1/f
tf = 2.5
N = Int(tf/dt)+1

####################################
## define training properties
####################################

num_train = 20
num_cross_validate = 5

# optimizer properties
max_iter = 2000
λ1 = 2e-3
λ2 = 1e-3

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

min_x = -0.0
max_x = 0.0

min_ẋ = 0.0
max_ẋ = 0.0

min_θ = pi
max_θ = pi

min_ω = 0.0
max_ω = 0.0

x_min = [min_x, min_θ, min_ẋ, min_ω]
x_max = [max_x, max_θ, max_ẋ, max_ω]

x0_perturb = [0, 0, 0, 0]
u0_perturb = [0.0]

u_min = [-0.0]
u_max = [0.0]

# equilibrium point
x_goal = [0.0, 0.0, 0.0, 0.0]
u_goal = [0.0]

y_goal_pixel = get_pixel_state(
    model_real, x_goal, resx, resy;
    sparse_image=sparse_data,
    ones_and_zeros=ones_and_zeros
)
y_goal_pixel = LinearPixelsToTorques.pixel_to_greyscale(y_goal_pixel, resx, resy)
num_y = length(y_goal_pixel)

#######################################################
## Create "teacher" controller and state estimator
#######################################################

# lqr controller
Q = Diagonal([100, 10, 1, 1])
R = Diagonal(1*ones(length(u_goal)))
Qf = 1000 .* Q

reference_generator = iLQRController(model_nominal, Q, Qf, R, dt, tf)
teacher_controller = TVLQRController(model_nominal, Q, Qf, R, dt, N; verbose=true)

# Kalman filter
C = [1 0 0 0; 0 1 0 0] # observability matrix (can only measure Cartpole angle not angular velocity)
R_cov = diagm([0.01, 0.01]) # measurement covariance
Q_cov = diagm([0.1, 0.1, 1, 1]) # process covariance

teacher_estimator = ExtendedKalmanFilter(model_nominal, C, Q_cov, R_cov, dt)

#######################################################
## Execute teacher controller and collect data
#######################################################

Random.seed!(1)
training_data, test_data = generate_swingup_data(model_real, reference_generator,
    teacher_controller, teacher_estimator, x_goal, u_goal,
    x_min, x_max, u_min, u_max, x0_perturb; num_train_ref=1,
    num_test_ref=0, num_rollouts_per_ref=100, tf=tf,
    dt=dt, resx=resx, resy=resy, visualize=false, sparse_image=sparse_data,
    process_noise_type=:dt, process_noise_factor=0.05,
    input_noise_type=:dt, input_noise_factor=0.0, measurement_noise_type=:encoder,
    measurement_noise_factor=[model_nominal.l, 2*pi]./600, ones_and_zeros=ones_and_zeros
);

jldsave(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_teacher_swingup_data.jld2");
    model_real, model_nominal, tf, dt, x_goal, u_goal, y_goal_pixel, training_data, test_data
)

#######################################################
## load training data
#######################################################

# load data
data = load(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_teacher_swingup_data.jld2"))
model_real = data["model_real"]
model_nominal = data["model_nominal"]
tf = data["tf"]
dt = data["dt"]
x_goal = data["x_goal"]
u_goal = data["u_goal"]
y_goal_pixel = data["y_goal_pixel"]
training_data = data["training_data"]
test_data = data["test_data"]

training_ind = 1:75
Y_pixels_training = training_data["X_pixel_training_traj"][training_ind][1:num_train]
X_real_training = training_data["X_real_training_traj"][training_ind][1:num_train]
X_est_training = training_data["X_est_training_traj"][training_ind][1:num_train]
X_ref_training = training_data["X_ref_training_traj"][training_ind][1:num_train]
U_real_training = training_data["U_real_training_traj"][training_ind][1:num_train]
U_est_training = training_data["U_est_training_traj"][training_ind][1:num_train]
U_ref_training = training_data["U_ref_training_traj"][training_ind][1:num_train]
x0_training = training_data["x0_training"][training_ind][1:num_train]
T = training_data["T"]

X_ref = X_ref_training[1]
U_ref = U_ref_training[1]

# extract cross validation data
cross_val_ind = 76:100
Y_pixels_cross_val = training_data["X_pixel_training_traj"][cross_val_ind][1:num_cross_validate]
X_real_cross_val = training_data["X_real_training_traj"][cross_val_ind][1:num_cross_validate]
X_est_cross_val = training_data["X_est_training_traj"][cross_val_ind][1:num_cross_validate]
U_real_cross_val = training_data["U_real_training_traj"][cross_val_ind][1:num_cross_validate]
U_est_cross_val = training_data["U_est_training_traj"][cross_val_ind][1:num_cross_validate]
x0_cross_val = training_data["x0_training"][cross_val_ind][1:num_cross_validate]

# extract test data
x0_test = test_data["x0_test"]

#######################################################
## visualize trajectory 
#######################################################

traj_num = 1 # <<--- change this number around

X = [SVector{4}(x) for x in X_real_training[traj_num]]
visualize!(vis, model_real, tf, X)

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

filename = joinpath(LinearPixelsToTorques.VIS_DIR, "example_cartpole_swingup.gif")
animate_pixels(model_real, Y_pixels_training[traj_num], filename, resx, resy; framerate=f)

#######################################################
## plot the trajectory 
#######################################################

x_real_hist = [x[1] for x in X]
θ_real_hist = [x[2] for x in X]

x_est_hist = [x[1] for x in X_est_training[traj_num]]
θ_est_hist = [x[2] for x in X_est_training[traj_num]]

x_ref = [x[1] for x in X_ref_training[traj_num]]
θ_ref = [x[2] for x in X_ref_training[traj_num]]

u_real_hist = [u[1] for u in U_real_training[traj_num]]
u_est_hist = [u[1] for u in U_est_training[traj_num]]
u_ref = [u[1] for u in U_ref_training[traj_num]]

fig1 = Plots.plot(T, x_real_hist, label="x (real)", xlabel="t", ylabel="state", title="Cartpole state", legend=:bottomright, lw=1.5)
Plots.plot!(fig1, T, x_est_hist,  label="x (estimated)", xlabel="t", ylabel="state", title="Cartpole state", lw=1.5)
Plots.plot!(fig1, T, x_ref, label="x (reference)", linestyle=:dash, lw=2)

fig2 = Plots.plot(T, θ_real_hist, label="θ (real)", xlabel="t", ylabel="state", title="Cartpole state", legend=:bottomright, lw=1.5)
Plots.plot!(fig2, T, θ_est_hist, label="θ (estimated)", xlabel="t", ylabel="state", title="Cartpole state", lw=1.5)
Plots.plot!(fig2, T, θ_ref, label="θ (reference)", linestyle=:dash, lw=2)

fig3 = Plots.plot(T[1:end-1], u_real_hist, label="u",
    xlabel="t", ylabel="control", title="cartpole Control", lw=2
)
Plots.plot!(fig3, T[1:end-1], u_est_hist, label="u (estimated)", lw=2)
Plots.plot!(fig3, T[1:end-1], u_ref, linestyle=:dash, lw=2,
    label="u (reference)"
)

display(fig1)
display(fig2)
display(fig3);

#######################################################
## Koopman embedding
#######################################################

function_list = ["state", "fourier", "chebyshev"]
states_list = [[true, true, true, true], [true, true, true, true], [true, true, true, true]
]
scale_list = [true, true, true]
order_list = [[0, 0], [1, 3], [2, 6]]
domain = [[-1.0, 0, -10, -20], [1.0, 2*pi, 10, 20]]

Z_est_training = [koopman_transform(X, function_list, states_list, order_list;
    scale_list=scale_list, domain=domain)
    for X in X_est_training
]
Zu_est_training = [bilinear_koopman_transform(X_est_training[i][1:end-1], U_est_training[i],
    function_list, states_list, order_list; scale_list=scale_list, domain=domain)
    for i in eachindex(X_est_training)
]

Z_est_cross_val = [koopman_transform(X, function_list, states_list, order_list;
    scale_list=scale_list, domain=domain)
    for X in X_est_cross_val
]
Zu_est_cross_val = [bilinear_koopman_transform(X_est_cross_val[i][1:end-1], U_est_cross_val[i],
    function_list, states_list, order_list; scale_list=scale_list, domain=domain)
    for i in eachindex(X_est_cross_val)
]

num_z = length(Z_est_training[1][1])
G = sparse(I, num_x, num_z)

println("Koopman embedding done!")

##################################################################
## learn L and C*L (inf-horizon Kalman gain and C*Kalman gain)
##################################################################

println("Creating LLS problem...")

RHS = concatenate_trajectories(Z_est_training; start_ind=2)
Zu_mat = concatenate_trajectories(Zu_est_training; start_ind=1)
Y_mat = concatenate_trajectories(Y_pixels_training; start_ind=2)

num_data = size(Zu_mat, 2)

println("Solving LLS problem...")

dynamics = Variable(num_z, num_z+num_u+num_z*num_u)
L = Variable(num_z, num_y)
d = Variable(num_z)

problem = minimize(sumsquares(dynamics*Zu_mat + L*Y_mat + d*ones(1, num_data) - RHS) +
    λ1*norm(vec(L), 1) + λ2*norm(vec(dynamics), 1) + λ2*norm(d, 1)
)
solve!(problem, () -> COSMO.Optimizer(max_iter=max_iter))

println("Extracting matrices...")

ABC = evaluate(dynamics)
A_hat = Matrix(ABC[:, 1:num_z])
B_hat = Matrix(ABC[:, num_z+1:num_z+num_u])
C_hat = Matrix(ABC[:, num_z+num_u+1:end])
L = evaluate(L)
d = evaluate(d)

C_hat_list = Matrix{Float64}[]
for i in 1:num_u
    C_i = C_hat[:, (i-1)*num_z+1:i*num_z]
    push!(C_hat_list, C_i)
end

println("Cross-validating...")

training_residual = norm(ABC*Zu_mat + L*Y_mat + d*ones(1, num_data) - RHS)

Zu_est_cross_val_mat = concatenate_trajectories(Zu_est_cross_val; start_ind=1)
RHS_cross_eval = concatenate_trajectories(Z_est_cross_val; start_ind=2)
Y_cross_val_mat = concatenate_trajectories(Y_pixels_cross_val; start_ind=2)
num_data_cross_val = size(Zu_est_cross_val_mat, 2)

test_residual = norm(ABC*Zu_est_cross_val_mat + L*Y_cross_val_mat + d*ones(1, num_data_cross_val) - RHS_cross_eval)

@show(training_residual)
@show(test_residual)

#######################################################
## save learned stuff
#######################################################

# save models
jldsave(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_koopman_swingup_example_data.jld2");
    model_real, model_nominal, A_hat, B_hat, C_hat_list, L, d, tf, dt
)

#######################################################
## load models
#######################################################

data = load(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_koopman_swingup_example_data.jld2"));
model_real = data["model_real"];
model_nominal = data["model_nominal"];
A_hat = data["A_hat"];
B_hat = data["B_hat"];
C_hat_list = data["C_hat_list"];
L = data["L"];
d = data["d"];
tf = data["tf"];
dt = data["dt"];

######################################
## Create Inf-horizon LQR controller
######################################

student_controller = deepcopy(teacher_controller)
student_estimator = BilinearKoopmanLuenbergerObserver(A_hat, B_hat, C_hat_list, L, d, G,
    function_list, states_list, order_list, scale_list, domain
)

println("student made!")

#######################################################
## execute lqr controller on real physics
#######################################################

X_ref_test = X_ref_training[1]
U_ref_test = U_ref_training[1]

update_controller!(teacher_controller, X_ref_test, U_ref_test, dt)
update_controller!(student_controller, X_ref_test, U_ref_test, dt)

x0 = x0_test[1]
u0 = [0.0]

Random.seed!(3)
X_real_eval, X_est_eval, X_pred_eval, U_real_eval, U_est_eval, Y_pixel_eval, T_eval, successful_sim = simulatewithpixelcontrollerandestimator(
    model_real, student_controller, student_estimator, x0, tf, dt; u0=u0,
    resx=resx, resy=resy, sparse_image=sparse_data,
    process_noise_type=:dt, process_noise_factor=0.05,
    input_noise_type=:dt, input_noise_factor=0.0, verbose=true,
    ones_and_zeros=ones_and_zeros
)

#######################################################
## visualize physical rollout results
#######################################################

X_eval = [SVector{4}(x) for x in X_real_eval]
visualize!(vis, model_real, T_eval[end], X_eval)

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

x_ref = [x[1] for x in X_ref_test]
θ_ref = [x[2] for x in X_ref_test]
v_ref = [x[3] for x in X_ref_test]
ω_ref = [x[4] for x in X_ref_test]

u_real_hist = [u[1] for u in U_real_eval]
u_est_hist = [u[1] for u in U_est_eval]
u_ref = [u[1] for u in U_ref_test]

fig1 = Plots.plot(T_eval, x_real_hist, label="x (real)", xlabel="t", ylabel="state", title="Cartpole state", legend=:bottomright, lw=1.5)
Plots.plot!(fig1, T_eval, x_est_hist,  label="x (estimated)", xlabel="t", ylabel="state", title="Cartpole state", lw=1.5)
Plots.plot!(fig1, T, x_ref, label="x (reference)", linestyle=:dash, lw=2)

fig2 = Plots.plot(T_eval, θ_real_hist, label="θ (real)", xlabel="t", ylabel="state", title="Cartpole state", legend=:bottomright, lw=1.5)
Plots.plot!(fig2, T_eval, θ_est_hist, label="θ (estimated)", xlabel="t", ylabel="state", title="Cartpole state", lw=1.5)
Plots.plot!(fig2, T_eval, θ_ref, label="θ (reference)", linestyle=:dash, lw=2)

fig3 = Plots.plot(T_eval, v_real_hist, label="v (real)", xlabel="t", ylabel="state", title="Cartpole state", legend=:bottomright, lw=1.5)
Plots.plot!(fig3, T_eval, v_est_hist, label="v (estimated)", xlabel="t", ylabel="state", title="Cartpole state", lw=1.5)
Plots.plot!(fig3, T_eval, v_ref, label="v (reference)", linestyle=:dash, lw=2)

fig4 = Plots.plot(T_eval, ω_real_hist, label="ω (real)", xlabel="t", ylabel="state", title="Cartpole state", legend=:bottomright, lw=1.5)
Plots.plot!(fig4, T_eval, ω_est_hist, label="ω (estimated)", xlabel="t", ylabel="state", title="Cartpole state", lw=1.5)
Plots.plot!(fig4, T_eval, ω_ref, label="ω (reference)", linestyle=:dash, lw=2)

fig5 = Plots.plot(T_eval[1:end-1], u_real_hist, label="u (real)",
    xlabel="t", ylabel="control", title="cartpole Control", lw=2
)
Plots.plot!(fig5, T_eval[1:end-1], u_est_hist, label="u (estimated)", lw=2)
Plots.plot!(fig5, T_eval[1:end-1], u_ref, linestyle=:dash, lw=2,
    label="u (reference)"
)

display(fig1)
display(fig2)
display(fig3)
display(fig4)
display(fig5);

################################
## save results
################################

jldsave(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_koopman_swingup_example_results.jld2");
    model_real, model_nominal, A_hat, B_hat, C_hat_list, L, d, tf,
    dt, X_real_eval, X_est_eval, X_pred_eval, U_real_eval, U_est_eval,
    Y_pixel_eval, X_ref_test, U_ref_test, T_eval
)