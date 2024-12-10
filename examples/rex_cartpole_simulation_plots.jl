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
using PGFPlotsX
using MeshCat
using GeometryBasics
using Colors
using CoordinateTransformations
import RobotDynamics as RD

if !isdir(LinearPixelsToTorques.DATA_DIR)
    mkdir(LinearPixelsToTorques.DATA_DIR)
end;
if !isdir(LinearPixelsToTorques.VIS_DIR)
    mkdir(LinearPixelsToTorques.VIS_DIR)
end;

function set_ref_cartpole!(vis, model::RExCartpole; 
    color=nothing, color2=nothing)
    dim = Vec(0.1, 0.3, 0.1)
    rod = Cylinder(Point3f0(0,-10,0), Point3f0(0,10,0), 0.01f0)
    cart = Rect3D(-dim/2, dim)
    hinge = Cylinder(Point3f0(-dim[1]/2,0,dim[3]/2), Point3f0(dim[1],0,dim[3]/2), 0.03f0)
    c1,c2 = [color,color2]

    pole = Cylinder(Point3f0(0,0,0),Point3f0(0,0,model.l),0.01f0)
    mass = HyperSphere(Point3f0(0,0,model.l), 0.05f0)
    setobject!(vis["rod"], rod, MeshPhongMaterial(color=colorant"grey"))
    setobject!(vis["cart","box"],   cart, MeshPhongMaterial(color=isnothing(color) ? colorant"green" : color))
    setobject!(vis["cart","hinge"], hinge, MeshPhongMaterial(color=isnothing(color) ? colorant"black" : color))
    setobject!(vis["cart","pole","geom","cyl"], pole, MeshPhongMaterial(color=c1))
    setobject!(vis["cart","pole","geom","mass"], mass, MeshPhongMaterial(color=c2))
    settransform!(vis["cart","pole"], Translation(0.75*dim[1],0,dim[3]/2))
end

####################################
## Simulation Parameters
####################################

sparse_image = true
stack_u = false
ones_and_zeros = false
learn_new_model = false

####################################
## define time properties
####################################

f = 60 # simulation frequency

dt = 1/f
tf = 12.5
N = Int(tf/dt)+1

T = range(0, tf, length=N)

####################################
## define image properties
####################################

# image resolution (res x res)
resx = 160
resy = 125

####################################
## make pendulum model
####################################

model = RExCartpole()

num_x_true = RD.state_dim(model)
num_u = RD.control_dim(model);
num_y = resx * resy

logocolors = Colors.JULIA_LOGO_COLORS

swingup_color = colorant"rgb(255,173,0)";
stabilizing_color = colorant"rgb(0,193,208)";
ref_color = colorant"rgba(196,18,48, 0.25)";
ref_plotting_color = colorant"rgb(196,18,48)";
real_color = colorant"rgb(0,0,0)"
println("colors made!")

####################################
## load stabilizing results
####################################

stabilizing_results = load(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_student_teacher_stabilizing_sweep_study_results.jld2"))

stabilizing_test_results = stabilizing_results["sweep_test_data"]
stabilizing_num_train_sweep = stabilizing_results["num_train_sweep"]
stabilizing_perf_median = stabilizing_results["stabilizing_perf_median"]
stabilizing_perf_min_quantile = stabilizing_results["stabilizing_perf_min_quantile"]
stabilizing_perf_max_quantile = stabilizing_results["stabilizing_perf_max_quantile"]
stabilizing_num_success = stabilizing_results["num_success"]
stabilizing_percent_success = stabilizing_num_success
X_est_stabilizing_trajs = stabilizing_test_results["100_training_traj"]["X_est_eval_traj"]

stabilizing_results_2 = load(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_student_teacher_stabilizing_sweep_study_training_results.jld2"))
L = stabilizing_results_2["sweep_student_controllers"]["150_training_traj"][2].L

####################################
## load swingup results
####################################

swingup_results = load(joinpath(LinearPixelsToTorques.DATA_DIR,"rex_cartpole_koopman_swingup_example_results.jld2"))

X_real_swingup = swingup_results["X_real_eval"]
X_est_swingup = swingup_results["X_est_eval"]
X_pred_swingup = swingup_results["X_pred_eval"]
U_real_swingup = swingup_results["U_real_eval"]
U_est_swingup = swingup_results["U_est_eval"]
Y_pixel_swingup = swingup_results["Y_pixel_eval"]
X_ref_test = swingup_results["X_ref_test"]
U_ref_test = swingup_results["U_ref_test"]
T_swingup = swingup_results["T_eval"]

####################################
## make MeshCat visualization
####################################

println("Booting up Meshcat")
vis = Visualizer()

setprop!(vis["/Background"], "top_color", colorant"rgb(255,255,255)")
setprop!(vis["/Background"], "bottom_color", colorant"rgb(255,255,255)")
render(vis)

#######################################################
## make MeshCat visualization for stabilization
#######################################################

set_ref_cartpole!(vis["ref"], model; color=ref_color, color2=ref_color)
visualize!(vis["ref"], model, SA[0., 0., 0., 0.])

set_mesh!(vis["real"], model; color=stabilizing_color, color2=stabilizing_color)
visualize!(vis["real"], model, SA[0., 0., 0., 0.])

#######################################################
## visualize a stabilizing trajectory 
#######################################################

traj_num = 2 # <<--- change this number around

X = [SVector{4}(x) for x in X_est_stabilizing_trajs[traj_num]]
# visualize!(vis["real"], model, tf, X; color=stabilizing_color, color2=stabilizing_color)

#######################################################
## choose knot points for visualization
#######################################################

knot_points = findall(x -> x in [0.0, 0.25, 1.0, 2.0, 3], T)

knot_point = knot_points[5] # <<--- change this number around

visualize!(vis["real"], model, X[knot_point])

#######################################################
## visualize an image from a knot point of trajectory
#######################################################

for knot_point in knot_points

    scene = construct_scene(resx, resy)
    LinearPixelsToTorques.visualize!(scene, model, X[knot_point])
    y_pixel_scene = LinearPixelsToTorques.get_scene(scene, sparse_image, ones_and_zeros)
    y_pixel = sparse_image ? sparsevec(y_pixel_scene) : vec(y_pixel_scene)
    y_greyscale_image = LinearPixelsToTorques.pixel_to_greyscale(y_pixel, resx, resy)
    display(y_greyscale_image)

end

####################################
## make MeshCat visualization
####################################

set_mesh!(vis["real"], model; color=swingup_color, color2=swingup_color)
visualize!(vis["real"], model, SA[0., 0., 0., 0.])

#######################################################
## visualize a swingup trajectory 
#######################################################

X = [SVector{4}(x) for x in X_real_swingup]
LinearPixelsToTorques.visualize!(vis["real"], model, 2.5, X; color=swingup_color)

##############################################################
## visualize an image from a specific knot point of swing-up
##############################################################

knot_points = findall(x -> x in [0.0, 0.2, 0.4, 0.6, 0.8, 1, 2.0, 2.5], T_swingup)

knot_point = knot_points[8] # <<--- change this number around

visualize!(vis["real"], model, X[knot_point])

##########################################
## visualize all knot points of swing-up
##########################################

for knot_point in knot_points

    scene = construct_scene(resx, resy)
    LinearPixelsToTorques.visualize!(scene, model, X[knot_point])
    y_pixel_scene = LinearPixelsToTorques.get_scene(scene, sparse_image, ones_and_zeros)
    y_pixel = sparse_image ? sparsevec(y_pixel_scene) : vec(y_pixel_scene)
    y_greyscale_image = LinearPixelsToTorques.pixel_to_greyscale(y_pixel, resx, resy)
    display(y_greyscale_image)

end

#######################################################
## extract states from swing-up trajectory
#######################################################

p_ref_hist = [x[1] for x in X_ref_test]
θ_ref_hist = [x[2] for x in X_ref_test]
u_ref_hist = [u[1] for u in U_ref_test]

p_real_hist = [x[1] for x in X_real_swingup]
θ_real_hist = [x[2] for x in X_real_swingup]
u_real_hist = [u[1] for u in U_real_swingup]

p_est_student_hist = [x[1] for x in X_est_swingup]
θ_est_student_hist = [x[2] for x in X_est_swingup]
u_est_student_hist = [u[1] for u in U_est_swingup]

#######################################################
## plot configurations from swing-up trajectory
#######################################################

fig1 = Plots.plot(T_swingup, [p_est_student_hist, p_real_hist, p_ref_hist], label=["Student" "Ground Truth" "Goal"],
    xlabel="Time (s)", ylabel="Cart Position (m)", lw=2, legend=:outerright,
    linecolor=[swingup_color real_color ref_plotting_color],
    linestyle=[:solid :dash :dot]
)

fig2 = Plots.plot(T_swingup, [θ_est_student_hist, θ_real_hist, θ_ref_hist], label=["Student" "Ground Truth" "Goal"],
    xlabel="Time (s)", ylabel="Pole Angle (rad)", lw=2, legend=:outerright,
    linecolor=[swingup_color real_color ref_plotting_color],
    linestyle=[:solid :dash :dot]
)

fig3 = Plots.plot(T_swingup[1:end-1], [u_est_student_hist, u_real_hist, u_ref_hist], label=["Student" "Ground Truth" "Goal"],
    xlabel="Time (s)", ylabel="Control", lw=2, legend=:outerright,
    linecolor=[swingup_color real_color ref_plotting_color],
    linestyle=[:solid :dash :dot]
)

display(fig1)
display(fig2)
display(fig3);

#######################################################
## make tikz of swing-up-configurations plots
#######################################################

lineopts = @pgf {no_marks, "very thick"}
position_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Cart Position (m)",
        legend_pos = "south east",
        xmin = 0,
        xmax = 2.5,
        ymin = -0.4,
        ymax = 0.4,
    },

    PlotInc({lineopts..., color=swingup_color, solid, thick},
        Coordinates(T_swingup, p_est_student_hist)),
    PlotInc({lineopts..., color=real_color, dashed, thick},
        Coordinates(T_swingup, p_real_hist)),
    PlotInc({lineopts..., color=ref_plotting_color, dotted, thick},
        Coordinates(T_swingup, p_ref_hist)),

)
pgfsave(joinpath(LinearPixelsToTorques.VIS_DIR, "fig5b_student_position_swingup_study.tikz"), position_plot, include_preamble=false)

angle_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Pole Angle (rad)",
        legend_pos = "south east",
        xmin = 0,
        xmax = 2.5,
        ymin = -0.5,
        ymax = 4,
    },

    PlotInc({lineopts..., color=swingup_color, solid, thick},
        Coordinates(T_swingup, θ_est_student_hist)),
    PlotInc({lineopts..., color=real_color, dashed, thick},
        Coordinates(T_swingup, θ_real_hist)),
    PlotInc({lineopts..., color=ref_plotting_color, dotted, thick},
        Coordinates(T_swingup, θ_ref_hist)),

)
pgfsave(joinpath(LinearPixelsToTorques.VIS_DIR, "fig5c_student_angle_swingup_study.tikz"), angle_plot, include_preamble=false)

control_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Control",
        legend_pos = "south east",
        xmin = 0,
        xmax = 2.5,
        ymin = -16,
        ymax = 10,
    },

    PlotInc({lineopts..., color=swingup_color, solid, thick},
        Coordinates(T_swingup[1:end-1], u_est_student_hist)),
    PlotInc({lineopts..., color=real_color, dashed, thick},
        Coordinates(T_swingup[1:end-1], u_real_hist)),
    PlotInc({lineopts..., color=ref_plotting_color, dotted, thick},
        Coordinates(T_swingup[1:end-1], u_ref_hist)),

    PGFPlotsX.Legend(["Student", "Ground Truth", "Goal"])
)
pgfsave(joinpath(LinearPixelsToTorques.VIS_DIR, "fig5d_student_control_swingup_study.tikz"), control_plot, include_preamble=false)

#################################
## plot the sweep study results
#################################

fig1 = Plots.plot(stabilizing_num_train_sweep, stabilizing_perf_median,
    ribbon=(stabilizing_perf_max_quantile .- stabilizing_perf_min_quantile)/2, fillalpha=0.1,
    label="Stabilizing", color=stabilizing_color,
    xlabel="Number of Training Trajectories", ylabel="Stabilization Error"
)
display(fig1)

fig2 = Plots.plot(stabilizing_num_train_sweep, stabilizing_percent_success,
    label="Stabilizing", color=stabilizing_color,
    xlabel="Number of Training Trajectories", ylabel="Successful Stabilizations (%)")
display(fig2)

#######################################################
## make tikz of plot
#######################################################

lineopts = @pgf {no_marks, "very thick"}
error_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of Training Trajectories",
        ylabel = "Stabilization Error",
        ymax = 20,
        ymode = "log"
    },

    PlotInc({lineopts..., "name_path=C", "cyan!10", "forget plot", solid, line_width=0.1},
        Coordinates(stabilizing_num_train_sweep, stabilizing_perf_min_quantile)),
    PlotInc({lineopts..., "name_path=D", "cyan!10", "forget plot", solid, line_width=0.1},
        Coordinates(stabilizing_num_train_sweep, stabilizing_perf_max_quantile)),
    PlotInc({lineopts..., "cyan!10", "forget plot"}, "fill between [of=C and D]"),
    PlotInc({lineopts..., color="cyan", solid, thick},
        Coordinates(stabilizing_num_train_sweep, stabilizing_perf_median)),
)
pgfsave(joinpath(LinearPixelsToTorques.VIS_DIR, "fig3b_student_error_sweep_study.tikz"), error_plot, include_preamble=false)

lineopts = @pgf {no_marks, "very thick"}
num_successful_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Number of Training Trajectories",
        ylabel = "Success Rate (\\%)",
        ymax = 110,
    },

    PlotInc({lineopts..., color="cyan", solid, thick},
        Coordinates(stabilizing_num_train_sweep, stabilizing_num_success)),
)
pgfsave(joinpath(LinearPixelsToTorques.VIS_DIR, "fig3c_student_num_success_sweep_study.tikz"), num_successful_plot, include_preamble=false)