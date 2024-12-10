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
using Makie
using Test
using ImageShow
using ControlSystems
using IterTools
using Convex
using COSMO
using ProgressMeter
using PGFPlotsX
import RobotDynamics as RD

if !isdir(LinearPixelsToTorques.DATA_DIR)
    mkdir(LinearPixelsToTorques.DATA_DIR)
end;
if !isdir(LinearPixelsToTorques.VIS_DIR)
    mkdir(LinearPixelsToTorques.VIS_DIR)
end;

####################################
## helper functions
####################################

function train_student(teacher_controller, teacher_estimator, training_data, training_ind; λ=1e-3, max_iter=5000)

    println("Extracting training data...")

    Y_pixels_training = training_data["X_pixel_training_traj"][training_ind]
    X_est_training = training_data["X_est_training_traj"][training_ind]
    U_est_training = training_data["U_est_training_traj"][training_ind]

    println("Creating LLS problem...")

    X_est_mat = concatenate_trajectories(X_est_training; start_ind=1, end_ind=1)
    U_mat = concatenate_trajectories(U_est_training; start_ind=1)

    num_data = size(X_est_mat, 2)

    RHS = concatenate_trajectories(X_est_training; start_ind=2)
    X_mat_1 = X_est_mat
    X_mat_2 = concatenate_trajectories(Y_pixels_training; start_ind=2)

    println("Solving LLS problem...")

    ALK_hat = Variable(num_x_true, num_x_true)
    L = Variable(num_x_true, num_y)
    d = Variable(num_x_true)

    problem = minimize(sumsquares(ALK_hat*X_mat_1 + L*X_mat_2 + d*ones(1, num_data) - RHS) + λ*norm(vec(L), 1))
    solve!(problem, () -> COSMO.Optimizer(max_iter=max_iter); silent=true)

    println("Extracting matrices...")

    ALK_hat = evaluate(ALK_hat)
    L = evaluate(L)
    d = evaluate(d)

    learning_residual = norm(ALK_hat*X_mat_1 + L*X_mat_2 + d*ones(1, num_data) - RHS)

    @show(learning_residual)

    println("Creating student controller...")

    student_controller = deepcopy(teacher_controller)
    student_estimator = LearnedFeedbackDynamicsLuenbergerObserver(ALK_hat, d, L);

    return student_controller, student_estimator

end

function test_student(student_controller, student_estimator, test_data, test_ind; tf=1.0, dt=0.01,
        resx=200, resy=125, sparse_data=true, process_noise_type=:gaussian,
        process_noise_factor=0.0, input_noise_type=nothing, input_noise_factor=0.0, verbose=false,
        termination_criteria=false
)

    N = Int(tf/dt)+1

    X0_test = test_data["x0_test"][test_ind]
    u0 = [0.0]

    X_real_eval_traj = [[X0_test[1] for _ in 1:N] for j in 1:length(X0_test)]
    X_est_eval_traj = [[X0_test[1] for _ in 1:N] for j in 1:length(X0_test)]
    X_pred_eval_traj = [[X0_test[1] for _ in 1:N] for j in 1:length(X0_test)]
    U_real_eval_traj = [[u0 for _ in 1:N-1] for j in 1:length(X0_test)]
    U_est_eval_traj = [[u0 for _ in 1:N-1] for j in 1:length(X0_test)]
    T_eval = range(0, tf, step=dt)
    successful_sim_list = [false for j in 1:length(X0_test)]

    @showprogress dt=1 desc="Testing current student..." for j in eachindex(X0_test)

        x0 = X0_test[j]

        X_real_eval, X_est_eval, X_pred_eval, U_real_eval, U_est_eval, _, T_eval, successful_sim = simulatewithpixelcontrollerandestimator(
            model_real, student_controller, student_estimator, x0, tf, dt; u0=u0,
            resx=resx, resy=resy, sparse_image=sparse_data,
            process_noise_type=process_noise_type, process_noise_factor=process_noise_factor,
            input_noise_type=input_noise_type, input_noise_factor=input_noise_factor, verbose=verbose,
            termination_criteria=termination_criteria
        )

        X_real_eval_traj[j] = X_real_eval
        X_est_eval_traj[j] = X_est_eval
        X_pred_eval_traj[j] = X_pred_eval
        U_real_eval_traj[j] = U_real_eval
        U_est_eval_traj[j] = U_est_eval
        successful_sim_list[j] = successful_sim

    end

    test_data = Dict(
        "X_real_eval_traj" => X_real_eval_traj,
        "X_est_eval_traj" => X_est_eval_traj,
        "X_pred_eval_traj" => X_pred_eval_traj,
        "U_real_eval_traj" => U_real_eval_traj,
        "U_est_eval_traj" => U_est_eval_traj,
        "T_eval" => T_eval,
        "successful_sim_list" => successful_sim_list
    )

    return test_data

end

function get_error_quantile(results; p=0.05)
    quantile_empty(x, i) = if isempty(x) 0 else quantile(x, i) end

    min_quant = quantile_empty(results, p)
    max_quant = quantile_empty(results, 1-p)
    return min_quant, max_quant
end

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

f = 60 # simulation frequency

dt = 1/f
tf = 2.5
test_tf = 10
N = Int(tf/dt)+1

####################################
## define training properties
####################################

num_train_sweep = Vector(10:10:150)
num_delay = 0

# optimizer properties
max_iter = 10000
λ_list = [5e-3, 5e-3, 5e-3, 1e-3, 1e-3,
    1e-4, 1e-4, 1e-4, 5e-5, 1e-5,
    1e-6, 1e-6, 1e-6, 1e-6, 1e-6
]

# image resolution (res x res)
resx = 160
resy = 125

####################################
## make pendulum model
####################################

model_nominal = RExCartpole()
model_real = RExCartpole(; mc=1.05 .* model_nominal.mc, mp=1.05 .* model_nominal.mp, l=model_nominal.l)

num_x_true = RD.state_dim(model_real)
num_u = RD.control_dim(model_real);
num_y = resx * resy

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

#######################################################
## Create "teacher" controller and state estimator
#######################################################

A_true = dfdx(model_real, x_eq, u_eq, dt)
B_true = dfdu(model_real, x_eq, u_eq, dt)

A_nominal = LinearPixelsToTorques.dfdx(model_nominal, x_eq, u_eq, dt)
B_nominal = LinearPixelsToTorques.dfdu(model_nominal, x_eq, u_eq, dt)

# lqr controller
Q = Diagonal(vcat(50, 100, 1 .* ones(num_x_true-2)))
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

# Random.seed!(1)
# training_data, test_data = generate_stabilizing_data(model_real, teacher_controller, teacher_estimator, x_min, x_max;
#     test_window=1.0, num_train=200, num_test=100, tf=tf,
#     dt=dt, resx=resx, resy=resy, visualize=false, sparse_image=sparse_data,
#     process_noise_type=:dt, process_noise_factor=0.05,
#     input_noise_type=nothing, input_noise_factor=0.0, measurement_noise_type=:encoder,
#     measurement_noise_factor=[model_nominal.l, 2*pi]./600, ones_and_zeros=ones_and_zeros
# );

# jldsave(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_teacher_stabilizing_data.jld2");
#     model_real, model_nominal, tf, dt, x_eq, u_eq, y_eq_pixel, training_data, test_data
# )

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
test_data = data["test_data"]

#######################################################
## train student controllers
#######################################################

training_params = Dict(
    "num_train_sweep" => num_train_sweep,
    "teacher_controller" => teacher_controller,
    "teacher_estimator" => teacher_estimator,
    "training_data" => training_data,
)

sweep_student_controllers = Dict()
iter = 1

@showprogress dt=1 desc="Training student controllers..." for num_train in num_train_sweep

    training_ind = 1:num_train
    λ = λ_list[iter]

    student_controller, student_estimator = train_student(
        teacher_controller, teacher_estimator, training_data, training_ind; λ=λ, max_iter=max_iter
    )

    key = "$num_train"*"_training_traj"
    sweep_student_controllers[key] = [student_controller, student_estimator]

    iter += 1

end

jldsave(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_student_teacher_stabilizing_sweep_study_training_results.jld2");
    training_params, sweep_student_controllers
)

#######################################################
## test student controllers
#######################################################

training_results = load(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_student_teacher_stabilizing_sweep_study_training_results.jld2"))

training_params = training_results["training_params"]
sweep_student_controllers = training_results["sweep_student_controllers"]

test_ind = 1:100

sweep_test_data = Dict()

@showprogress dt=1 desc="Testing student controllers..." for num_train in num_train_sweep
    
    key = "$num_train"*"_training_traj"

    student_controller, student_estimator = sweep_student_controllers[key]

    student_test_data = test_student(
        student_controller, student_estimator, test_data, test_ind;
        tf=test_tf, dt=dt, resx=resx, resy=resy, sparse_data=sparse_data,
        process_noise_type=:dt, process_noise_factor=0.05, input_noise_type=:dt,
        input_noise_factor=0.0, verbose=false, termination_criteria=(x) -> any(abs.(x[1:2]) .> [0.5, deg2rad(30)])
    )

    sweep_test_data[key] = student_test_data

end

jldsave(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_student_teacher_stabilizing_sweep_study_test_results.jld2");
    training_params, sweep_student_controllers, sweep_test_data
)

####################################
## Import test results
####################################

test_results = load(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_student_teacher_stabilizing_sweep_study_test_results.jld2"))

training_params = test_results["training_params"]
sweep_student_controllers = test_results["sweep_student_controllers"]
sweep_test_data = test_results["sweep_test_data"]

#################################################
## Post-process test results
#################################################

stabilizing_perf_median = zeros(length(num_train_sweep))
stabilizing_perf_min_quantile = zeros(length(num_train_sweep))
stabilizing_perf_max_quantile = zeros(length(num_train_sweep))
num_success = zeros(length(num_train_sweep))

@showprogress dt=1 desc="Postprocessing..." for i in 1:length(num_train_sweep)

    num_train = num_train_sweep[i]
    key = "$num_train"*"_training_traj"

    # no structure
    student_controller, student_estimator = sweep_student_controllers[key]
    student_test_data = sweep_test_data[key]

    X_real_eval_traj = student_test_data["X_real_eval_traj"]
    successful_sim_list = student_test_data["successful_sim_list"]

    stabilizing_perf = []

    for i in 1:100

        if successful_sim_list[i]
            X_end = mean(X_real_eval_traj[i][end-(N÷5)+1:end])
        else
            X_end = X_real_eval_traj[i][end]
        end

        push!(stabilizing_perf, norm(X_end - x_eq))

        if successful_sim_list[i] && any(abs.(X_end - x_eq)[1:2] .> [model_nominal.l/2, deg2rad(2)])
            successful_sim_list[i] = false
        end

    end

    min_quant, max_quant = get_error_quantile(filter(!isnan, stabilizing_perf); p=0.05)
    median_perf = median(filter(!isnan, stabilizing_perf))

    stabilizing_perf_median[i] = median_perf
    stabilizing_perf_min_quantile[i] = min_quant
    stabilizing_perf_max_quantile[i] = max_quant
    num_success[i] = sum(successful_sim_list)

    println("$num_train data:\n")
    println("Median perf = ", median_perf)
    println("5% Min quantile = ", min_quant)
    println("95% Max quantile = ", max_quant)
    println("Num success = ", sum(successful_sim_list))
    println("")
    
end

#################################################
## Plot results
#################################################

fig1 = Plots.plot(num_train_sweep, stabilizing_perf_median, yaxis=:log,
    label="Unstructured Training",
    xlabel="Number of Training Trajectories", ylabel="Stabilization Error",
    title="Stabilization Performance (Median Error)"
)

fig2 = Plots.plot(num_train_sweep, num_success, label="Unstructured Training",
    xlabel="Number of Training Trajectories", ylabel="# of Successful Stabilizations",
    title="Stabilization Performance (# of Successes)"
)

display(fig1)
display(fig2)

#######################################################
## Save results
#######################################################

jldsave(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_student_teacher_stabilizing_sweep_study_results.jld2");
    num_train_sweep, stabilizing_perf_median, stabilizing_perf_min_quantile,
    stabilizing_perf_max_quantile, num_success, sweep_test_data, training_params
)