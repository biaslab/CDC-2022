module NARMAXExperiments

using Distributions
using LinearAlgebra
using Random
using Optim: optimize, LBFGS
using ForwardDiff
using Rocket, GraphPPL, ReactiveMP
using Parameters


include("data.jl")
include("nonlinear_node.jl")
include("polynomial.jl")
include("utils.jl")
include("models.jl")
include("NARMAX_fl.jl")

end