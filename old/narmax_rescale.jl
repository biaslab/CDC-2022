using Pkg
Pkg.add("GraphPPL", "Rocket", "ReactiveMP", "Distributions", "LinearAlgebra", "Random", "JLD", "Parameters", "ProgressMeter", "CSV", "DataFrames")

# import packages
using GraphPPL
using Rocket
using ReactiveMP
using Distributions
using LinearAlgebra
using Random
using JLD
using Parameters
using CSV
using DataFrames
import ProgressMeter
import ReactiveMP.messageout

include("../helpers/polynomial.jl")
include("../helpers/ar_extension.jl")
include("../helpers/nonlinear_node.jl")