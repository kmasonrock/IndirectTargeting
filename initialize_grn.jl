using Graphs
using MetaGraphsNext # Needed to assign edge labels 
using QuasiMonteCarlo # Needed to do Latin Hypercube Sampling
using Configurations
using Dates
using AdaptiveKDTrees.KNN
using StatsBase
using Random
using PreallocationTools
using Distributed
using OrdinaryDiffEq
using UnPack
using ForwardDiff
using CairoMakie
using GraphMakie
using JuMP
using SymEngine_jll
using SymEngine
using SymEngine: CMapBasicBasic, CDenseMatrix
using MacroTools: @capture, prewalk, MacroTools
using RuntimeGeneratedFunctions
import Ipopt, HSL_jll
using IterTools
using DiffEqCallbacks
using LinearAlgebra
using Combinatorics
using JLD2
using TOML
using Glob
#=
The following code is based of the paper titled
"Generic properties of random gene regulatory networks", by
Zhiyuan Li, Simone Bianco, Zhaoyang Zhang, Chao Tang (2015)

The idea is as follows:

Given a network of N nodes, a connectivity constant K, and a
inhibinatory regulation constant ϵ, assign a link between two nodes 
with probability K/N, and label this link as inhibitory or activing
with probability ϵ and 1-ϵ respectfully.

For dynamics, we use a combination of hill functions that represent
inhibition and activation. The idea is that we generate a connected
network, and then do a parameter space search by randomizing K and n
(See GRNParams structure for details on these terms) until we find valid
dynamics where all genes have a steady state.

NOTE -- These dynamics have been known to generate limit cycles and chaos.
Steady states are indeed possible, but the findings in the above paper show
that they're harder to generate with the selected dynamics. If you want this
behaviour, refer to the "Strong Reciprocal" version of the dynamics (which I
have not coded here due to being too lazy to figure out how the authors chose
weights for each edge).
=#

## STRUCTURES ##

# NOTE:  Some structures need to be mutable in order for the log_space kwarg to
#        effectively change the values for the logarithmic system.


#Structure for all of the Network Gen Parameters
@option "networkgen_params" struct NetworkGenParams
    N::Int64 = 5
    K::Float64 = 2
    ϵ::Float64 = 0.5
end

# Structure for the Dynamics Parameters
@option "dynamic_params" struct DynamicsParams
    G::MetaGraph
    activators::Vector{Vector{Int}}
    inhibitors::Vector{Vector{Int}}
    K::Vector{Float64} = vec(10 .^ (QuasiMonteCarlo.sample(nv(G), -3, 1, LatinHypercubeSample())))#Dissociation Constant -- uniformly Log sampled between 1e-3 and 10 via Latin Hypercube Sampling
    n::Vector{Float64} = rand(1:10, nv(G)).*1.0 #Polymerization and Cooperative Binding of Transcription Factors -- Uniform sampled between 1 - 10
    Kn::Vector{Float64} = K.^n
    w::Vector{Float64} = [indegree(G,v) == 0 ? 0 : 1/(indegree(G,v)) for v in vertices(G)]
    dummy::Vector{DiffCache{Vector{Float64}, Vector{Float64}}} = [DiffCache(zeros(nv(G))) for i in 1:5]
    dyn_abstol::Float64 = 1e-10
    dyn_reltol::Float64 = 1e-8
end

#Structure for the Optimization Algorithm
@option "optimization_params" struct OptimizationParams
    iter_limit::Int64 = 700
    closest_time::Vector{Float64} = [0.0, 10.0]
    integrate_time::Vector{Float64} = [0.0, 5000.0]
    target_states::Vector{Vector{Float64}}
    target_nodes::Vector{Int64}
    ϵ_max::Float64 = 0.01
    ϵ_min::Float64 = 0.001
    rounding_digits::Int64 = 1
    time_increment::Float64 = 0.1
    opt_abstol::Float64 = 1e-10
    opt_reltol::Float64 = 1e-8
    opt_maxiters::Int64 = 100_000
end

#Structure for all of the general parameters
@option "universal_params" struct UniversalParams
    log_space::Bool = false
    net_gen::NetworkGenParams
    opt::OptimizationParams
end

struct ModelParams
    func::ODEFunction
    J::Matrix{Float64}
    n::Int64
    dyn_p::DynamicsParams
end

struct SaveObject
    up::UniversalParams
    dyn::DynamicsParams
    stable::Vector{Float64}
end

RuntimeGeneratedFunctions.init(@__MODULE__)
include("scripts/gene_regulatory/grn_generation.jl")
include("scripts/gene_regulatory/grn_function.jl")
include("scripts/indirect_targeting_algorithm/perturbations.jl")
include("scripts/indirect_targeting_algorithm/FastMTK.jl")
