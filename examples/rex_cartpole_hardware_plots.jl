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
using Colors
using PGFPlotsX
using ProgressMeter
using CairoMakie
using Colors

const MOI = Convex.MOI

if !isdir(LinearPixelsToTorques.DATA_DIR)
    mkdir(LinearPixelsToTorques.DATA_DIR)
end;
if !isdir(LinearPixelsToTorques.VIS_DIR)
    mkdir(LinearPixelsToTorques.VIS_DIR)
end;

#######################################################
## plot colors
#######################################################

logocolors = Colors.JULIA_LOGO_COLORS

color_student = colorant"rgb(0,193,208)"
color_teacher = colorant"rgb(0,0,0)"
color_goal = colorant"rgb(196,18,48)"
print("colors made")

####################################
## import 1st data
####################################

data_file_name = "2e-2_reg_assume_feedback_20_traj_stabilization_results.npz";
full_data_file_path = joinpath(LinearPixelsToTorques.DATA_DIR, data_file_name);
data = npzread(full_data_file_path);

resy, resx = size(data["frames"][:, :, 1])

X_est_teacher_mat = data["est_states_encoder"]
X_est_student_mat = data["est_states_pixel"]

U_teacher_mat = reshape(data["controls_encoder"], 1, length(data["controls_encoder"]))
U_student_mat = reshape(data["controls_pixel"], 1, length(data["controls_pixel"]))

Y_pixels_list = [vec(convert(Matrix{Float64}, data["frames"][:, :, i])) for i in 1:size(data["frames"], 3)]
Y_pixels_mat = sparse(reduce(hcat, Y_pixels_list))

X_est_teacher_traj = [X_est_teacher_mat[:, i] for i in 1:size(X_est_teacher_mat, 2)];
X_est_student_traj = [X_est_student_mat[:, i] for i in 1:size(X_est_student_mat, 2)];

U_teacher_traj = [U_teacher_mat[:, i] for i in 1:size(U_teacher_mat, 2)];
U_student_traj = [U_student_mat[:, i] for i in 1:size(U_student_mat, 2)];

Y_pixels_traj = [Y_pixels_mat[:, i] for i in 1:size(Y_pixels_mat, 2)];

data_file_name = "learned_hardware_lo_matrices_2e-2_reg_assume_feedback_20_traj.npy";
full_data_file_path = joinpath(LinearPixelsToTorques.DATA_DIR, data_file_name);
data = npzread(full_data_file_path);

L = data["L"]

#######################################################
## save data in julia format
#######################################################

# save models
jldsave(joinpath(LinearPixelsToTorques.DATA_DIR, "hardware_stabilization_results.jld2");
    resy, resx, X_est_teacher_traj, X_est_student_traj, U_teacher_traj, U_student_traj, Y_pixels_traj,
    L
)

#######################################################
## learning setup
#######################################################

model_nominal = RExCartpole()

num_x = RD.state_dim(model_nominal)
num_u = RD.control_dim(model_nominal);

K = 0.2 .* [-3.29489, -19.36664, -4.46195, -2.79850/1.5]'

#######################################################
## load models
#######################################################

data = load(joinpath(LinearPixelsToTorques.DATA_DIR, "hardware_stabilization_results.jld2"));
resy = data["resy"]
resx = data["resx"]
num_y = resx * resy

# extract trajectory data
X_est_teacher_traj = data["X_est_teacher_traj"]
X_est_student_traj = data["X_est_student_traj"]
U_teacher_traj = data["U_teacher_traj"]
U_student_traj = data["U_student_traj"]
Y_pixels_traj = data["Y_pixels_traj"]
T = 0:1/60:(length(X_est_teacher_traj)-1)/60

L = data["L"]

###################################################
## truncate data to start at first perturbation
###################################################

truncate_idx_start = findall(x -> x in [15 - 15/60], T)[1]
truncate_idx_end = findall(x -> x in [25 - 15/60], T)[1]

X_est_teacher_traj = data["X_est_teacher_traj"][truncate_idx_start:truncate_idx_end]
X_est_student_traj = data["X_est_student_traj"][truncate_idx_start:truncate_idx_end]
U_teacher_traj = data["U_teacher_traj"][truncate_idx_start:truncate_idx_end]
U_student_traj = data["U_student_traj"][truncate_idx_start:truncate_idx_end]
Y_pixels_traj = data["Y_pixels_traj"][truncate_idx_start:truncate_idx_end]
T = 0:1/60:(length(X_est_teacher_traj)-1)/60

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

X = [SVector{4}(x) for x in X_est_teacher_traj]

visualize!(vis, model_nominal, length(X)÷60, X)

#######################################################
## animate trajectory
#######################################################

Y_pixels_traj_gray = [LinearPixelsToTorques.pixel_to_greyscale(y, resx, resy) for y in Y_pixels_traj]

filename = joinpath(LinearPixelsToTorques.VIS_DIR, "hardware_cartpole_perturbation_rejection.mp4")
scene = construct_scene(resx, resy)
frames = 1:length(Y_pixels_traj_gray)

p = Progress(length(frames), 1, "Creating animation...")

record(scene, filename, frames, framerate=60) do i

    y_i = Y_pixels_traj_gray[i]
    y_i[y_i .< 0.0] .= 0.0
    y_i[y_i .> 1.0] .= 1.0

    Makie.image!(scene, rotr90(y_i))

    next!(p)

end

#######################################################
## visualize an images of trajectory
#######################################################

knot_point = findall(x -> x in [6 + 16/60], T)[1]

y_pixel = Y_pixels_traj[knot_point]
y_greyscale_image = LinearPixelsToTorques.pixel_to_greyscale(y_pixel, resx, resy)
display(y_greyscale_image)

#######################################################
## visualize an images from knot points of trajectory
#######################################################

knot_points = findall(x -> x in [0, 0.5, 1.5, 5.5, 6, 7, 9.5, 10], T)

knot_point = knot_points[4] # <<--- change this number around

for knot_point in knot_points

    scene = construct_scene(resx, resy)
    y_pixel = Y_pixels_traj[knot_point]
    y_greyscale_image = LinearPixelsToTorques.pixel_to_greyscale(y_pixel, resx, resy)
    display(y_greyscale_image)

end

#######################################################
## extract states from trajectories
#######################################################

p_est_teacher_hist = [x[1] for x in X_est_teacher_traj]
θ_est_teacher_hist = [x[2] for x in X_est_teacher_traj]
u_est_teacher_hist = [u[1] for u in U_teacher_traj]

p_est_student_hist = [x[1] for x in X_est_student_traj]
θ_est_student_hist = [x[2] for x in X_est_student_traj]
u_est_student_hist = [u[1] for u in U_student_traj]

p_goal = zeros(length(T))
θ_goal = zeros(length(T))
u_goal = zeros(length(T))

#######################################################
## plot configurations from trajectories
#######################################################

fig1 = Plots.plot(T, [p_est_student_hist, p_est_teacher_hist, p_goal], label=["Student" "Teacher (Ground Truth)" "Goal"],
    xlabel="Time (s)", ylabel="Cart Position (m)", lw=2, legend=:outerright,
    linecolor=[color_student color_teacher color_goal],
    linestyle=[:solid :dash :dot]
)

fig2 = Plots.plot(T, [-θ_est_student_hist, -θ_est_teacher_hist, θ_goal], label=["Student" "Teacher (Ground Truth)" "Goal"],
    xlabel="Time (s)", ylabel="Pole Angle (rad)", lw=2, legend=:outerright,
    linecolor=[color_student color_teacher color_goal],
    linestyle=[:solid :dash :dot]
)

fig3 = Plots.plot(T, [u_est_student_hist, u_est_teacher_hist, u_goal], label=["Student" "Teacher" "Goal"],
    xlabel="Time (s)", ylabel="Control", lw=2, legend=:outerright,
    linecolor=[color_student color_teacher color_goal],
    linestyle=[:solid :dash :dot]
)

display(fig1)
display(fig2)
display(fig3);

#######################################################
## Presentation plots
#######################################################

fig1 = Plots.plot(T, [10 .* p_est_student_hist, 10 .* p_est_teacher_hist],
    xlabel="Time (s)", ylabel="Cart Position (cm)", lw=2,
    linecolor=[colorant"rgb(0, 176, 80)" colorant"rgb(255, 192, 0)"],
    linestyle=[:solid :dash], background_color=:transparent, xgridcolor=:gray, ygridcolor=:gray,
    xgridwidth=2, ygridwidth=2, x_guidefontcolor=:white, y_guidefontcolor=:white,
    xtickfontcolor=:white, ytickfontcolor=:white, legend=false, legendfontsize=12,
    x_foreground_color_border=:white, y_foreground_color_border=:white, xtickfontsize=12,
    ytickfontsize=12, x_guidefontsize=12, y_guidefontsize=12, titlefontsize=12,
    x_foreground_color_axis=:white, y_foreground_color_axis=:white,
    xlims=(0, 10), ylims=(-2, 3), dpi=100
)
display(fig1)
savefig(fig1, joinpath(LinearPixelsToTorques.VIS_DIR, "presentation_position_hardware_plot.svg"))

fig2 = Plots.plot(T, [rad2deg.(-θ_est_student_hist), rad2deg.(-θ_est_teacher_hist)],
    label=["Student" "Teacher (Ground Truth)" "Goal"],
    xlabel="Time (s)", ylabel="Pole Angle (deg)", lw=2,
    linecolor=[colorant"rgb(0, 176, 80)" colorant"rgb(255, 192, 0)"], linestyle=[:solid :dash],
    background_color=:transparent, xgridcolor=:gray, ygridcolor=:gray,
    xgridwidth=2, ygridwidth=2, x_guidefontcolor=:white, y_guidefontcolor=:white,
    xtickfontcolor=:white, ytickfontcolor=:white, legend=false,
    x_foreground_color_border=:white, y_foreground_color_border=:white, xtickfontsize=12,
    ytickfontsize=12, x_guidefontsize=12, y_guidefontsize=12, titlefontsize=12,
    x_foreground_color_axis=:white, y_foreground_color_axis=:white,
    xlims=(0, 10), ylims=(-15, 10), dpi=100
)
display(fig2)
savefig(fig2, joinpath(LinearPixelsToTorques.VIS_DIR, "presentation_angle_hardware_plot.svg"))

#######################################################
## make tikz of plots
#######################################################

lineopts = @pgf {no_marks, "very thick"}
position_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Cart Position (m)",
        legend_pos = "south east",
        xmin=0,
        xmax=10,
        ymin=-0.2,
        ymax=0.3
    },

    PlotInc({lineopts..., color=color_student, solid, thick},
        Coordinates(T, p_est_student_hist)),
    PlotInc({lineopts..., color=color_teacher, dashed, thick},
        Coordinates(T, p_est_teacher_hist)),
    PlotInc({lineopts..., color=color_goal, dotted, thick},
        Coordinates(T, p_goal)),
)
pgfsave(joinpath(LinearPixelsToTorques.VIS_DIR, "fig7b_position_hardware_plot.tikz"), position_plot, include_preamble=false)

angle_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Pole Angle (rad)",
        legend_pos = "south east",
        xmin=0,
        xmax=10,
        ymin=-0.15,
        ymax=0.25
    },

    PlotInc({lineopts..., color=color_student, solid, thick},
        Coordinates(T, -θ_est_student_hist)),
    PlotInc({lineopts..., color=color_teacher, dashed, thick},
        Coordinates(T, -θ_est_teacher_hist)),
    PlotInc({lineopts..., color=color_goal, dotted, thick},
        Coordinates(T, θ_goal)),
)
pgfsave(joinpath(LinearPixelsToTorques.VIS_DIR, "fig7c_angle_hardware_plot.tikz"), angle_plot, include_preamble=false)

control_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Control",
        legend_pos = "south east",
        xmin=0,
        xmax=10,
        ymin=-0.6,
        ymax=0.5
    },

    PlotInc({lineopts..., color=color_student, solid, thick},
        Coordinates(T, u_est_student_hist)),
    PlotInc({lineopts..., color=color_teacher, dashed, thick},
        Coordinates(T, u_est_teacher_hist)),
    PlotInc({lineopts..., color=color_goal, dotted, thick},
        Coordinates(T, u_goal)),

    PGFPlotsX.Legend(["Student", "Teacher", "Goal"])
)
pgfsave(joinpath(LinearPixelsToTorques.VIS_DIR, "fig7d_control_hardware_plot.tikz"), control_plot, include_preamble=false)