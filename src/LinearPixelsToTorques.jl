module LinearPixelsToTorques

greet() = print("Hello World!")

using LinearAlgebra
using SparseArrays
using StaticArrays
using OSQP
using ForwardDiff, FiniteDiff
using Statistics
using ProgressMeter
using Polynomials
using JLD2
using RobotZoo
using Rotations
using GeometryBasics
using CoordinateTransformations
using Colors
using MeshCat
using CairoMakie
using Distributions
using Convex
using COSMO
using Base.Threads
using BlockDiagonals
using IterTools
import Random

const MOI = Convex.MOI

import RobotDynamics
import RobotDynamics as RD

abstract type AbstractController end
abstract type AbstractEstimator end

export LinearModel, BilinearModel, discrete_dynamics, discrete_jacobian

export generate_stabilizing_data, generate_stabilizing_trajectory, generate_stabilizing_data_no_images

export generate_swingup_data, generate_swingup_trajectory, generate_swingup_data_no_images

export AbstractController, getmeasurement, getcontrol

export AbstractEstimator, estimator_step

export InfLQRController, inf_horizon_ricatti

export iLQRController, generate_trajectory, ilqr_solve

export TVLQRController, update_controller!

export simulatewithpixelcontrollerandestimator, simulatewithcontrollerandestimator,
    simulatewithpixelcontroller, simulatewithcontroller, simulatewithestimator,
    simulate, dynamics_rk4, dfdx, dfdu

export construct_scene, get_greyscale, get_rgb, get_pixel_state, visualize!,
    pixel_to_greyscale, animate_pixels, filter_zero_rows_matrix, get_pixel_state_og

export learn_bilinear_model, build_data_matrices, concatenate_trajectories, concatenate_trajectory

export KalmanFilter, LearnedFeedbackDynamicsLuenbergerObserver,
    BilinearKoopmanLuenbergerObserver, reset_covariance!,
    update_covariance!

export ExtendedKalmanFilter, reset_covariance!, update_covariance!

export koopman_transform, bilinear_koopman_transform, state_transform,
    state, sine, cosine, fourier, hermite, chebyshev, monomial

export visualize!, set_mesh!, waypoints!, traj3!

export RExCartpole

include("RExCartpole.jl")
include("concatenate_trajectories.jl")
include("BilinearModel.jl")
include("LinearModel.jl")
include("dynamics_and_simulation.jl")
include("lqr_controllers.jl")
include("tvlqr_controllers.jl")
include("ilqr_controller.jl")
include("renderer.jl")
include("meshcat_visualizations.jl")
include("stabilizing_data_generator.jl")
include("swingup_data_generator.jl")
include("luenberger_observers.jl")
include("extended_kalman_filters.jl")
include("koopman_basis.jl")

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const TEST_DIR = joinpath(@__DIR__, "..", "test")
const VIS_DIR = joinpath(@__DIR__, "..", "visualization")
const DATA_DIR = joinpath(@__DIR__, "..", "data")

end