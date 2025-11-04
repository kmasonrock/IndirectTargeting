as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...))

function is_fixedpoint(func, vals, du, p, thresh = 1e-5)
    func(du,vals,p,nothing)
    return all(x -> abs(x) < thresh, du)
end

function is_stable(func, vals, du, p)
    J = ForwardDiff.jacobian((_du, _u) -> func(_du, _u, p, nothing), du, vals)
    try
        return maximum(real(eigvals(J))) < 0
    catch e
        return false
    end
end

function random_gene_network(N, K, ϵ; dynamics_params = nothing, force_self_loops = false)
    
    inhibitor = Symbol("inhibitor")
    activator = Symbol("activator")
    

    connection_prob = K/N

    vertices_description = [i => i for i in 1:N]
    edges_description = Vector{Pair{Tuple{Int64, Int64}, Symbol}}(undef, 0)

    G = SimpleDiGraph(N)

    for i in vertices(G)
        for j in vertices(G)
            if i != j
                if rand() < connection_prob
                    add_edge!(G, i, j)
                    if rand() < ϵ
                        push!(edges_description, (i,j) => inhibitor)
                    else
                        push!(edges_description, (i,j) => activator)
                    end
                end
            end
        end
    end

    G = MetaGraph(G, vertices_description, edges_description, "regulatory network")
    all_edges = collect(edges(G))
    inhib_list = []
    activ_list = []
    for v in vertices(G)
        vertex_mask = findall([i.dst == v for i in collect(edges(G))])
        vertex_edges = all_edges[vertex_mask]
        inhibitor_mask = findall([G[i.src, v] == :inhibitor for i in vertex_edges])
        activator_mask = findall([G[i.src, v] == :activator for i in vertex_edges])
        inhibitors = [vertex_edges[i].src for i in inhibitor_mask]
        activators = [vertex_edges[i].src for i in activator_mask]

        push!(inhib_list, inhibitors)
        push!(activ_list, activators)
    end

    if isnothing(dynamics_params)
        p = DynamicsParams(G = G, activators = activ_list, inhibitors = inhib_list)
    else
        p = DynamicsParams(G = G, activators = activ_list, inhibitors = inhib_list, K = ones(nv(G))*dynamics_params[1], n = ones(nv(G))*dynamics_params[2])
    end
    u = QuasiMonteCarlo.sample(N, 0, 1, LatinHypercubeSample())

    return G, p, u
end

function make_grn(N, K, ϵ, func; dynamics_params = nothing, force_self_loops = false)
    expression_thresh = 1e-5
    expressed_thresh = 3
    sol_atol = 1e-8
    sol_rtol = 1e-6
    save_step = 0.1
    integrate_time = [0 5000]

    G, dp, u = random_gene_network(N, K, ϵ; dynamics_params = dynamics_params, force_self_loops = force_self_loops)

    u = vec(u)

    if !is_connected(G)
        return nothing
    end

    du = zeros(N)
    prob = ODEProblem(func, u, integrate_time, dp)
    sol = solve(prob, Vern9(), abstol = sol_atol, reltol = sol_rtol, saveat = save_step)
    if (!is_fixedpoint(func, sol[end], du, dp)  | !is_stable(func, sol[end], du, dp)) 
        return nothing
    else
        fp_list, ideal_pairs, thresholds = search_for_fp(dp, sol[end], func)
        if isempty(ideal_pairs)
            return nothing
        else
            op = OptimizationParams(target_states = [], target_nodes = [])
            return G, dp, op, fp_list, ideal_pairs, thresholds
        end
    end

    #println("Fixed Point and Stablity Passed!")

    target_genes = Vector{Int64}(undef, 0)
    target_states = Vector{Vector{Float64}}(undef,0)
    for gene in vertices(G)
        new_u0 = copy(sol[end])
        if round.(new_u0[gene]; digits = 4) >= 0.5
            new_u0[gene] = 0
        else
            new_u0[gene] = 1
        end
        new_prob = ODEProblem(func, new_u0, [0 10000], dp)
        new_sol = solve(new_prob, Vern9(), abstol = sol_atol, reltol = sol_rtol, saveat = save_step)
        if all(x -> x == true, round.(new_sol[end][Not(gene)]; digits = 2) .== round.(sol[end][Not(gene)]; digits = 2)) & is_stable(func, new_sol[end], du, dp) & is_fixedpoint(func, new_sol[end], du, dp)
            if (new_sol[end][gene] != sol[end][gene]) & (abs(new_sol[end][gene] - sol[end][gene]) > 0.5)
                push!(target_genes, gene)
                push!(target_states, new_sol[end])
            end
        end
    end
    #println(length(target_genes))
    if isempty(target_genes)
        return nothing
    else
        op = OptimizationParams(target_states = target_states, target_nodes = target_genes)
        return G, dp, sol[end], op
    end
end

function generate_grns(num_graphs, save_loc, func; dynamics_params = nothing, toml_file_name = "GRN_NetworkGen_", force_self_loops = false, ngp_custom = nothing)
    mkpath(save_loc)
    
    if isnothing(ngp_custom)
        ngp = NetworkGenParams()
    else
        ngp = NetworkGenParams(ngp_custom[1], ngp_custom[2], ngp_custom[3])
    end

    @unpack N, K, ϵ = ngp

    completed_graphs = 0
    local up, dp

    iter = 0
    
    fp_list = []
    ideal_pairs_list = []
    dyn_params = []
    up_list = []

    while completed_graphs < num_graphs
        
        vals = make_grn(N, K, ϵ, func; dynamics_params = dynamics_params, force_self_loops = force_self_loops)
        if isnothing(vals)
            continue
        else
            G, dp, op, fps, ideal_pairs, thresholds = vals
            push!(fp_list, fps)
            push!(ideal_pairs_list, ideal_pairs)
            up = UniversalParams(net_gen = ngp, opt = op)
            push!(dyn_params, dp)
            push!(up_list, up)
            completed_graphs += 1
            DataDict = Dict()
            DataDict["fps"] = fps
            DataDict["ideal_pairs"] = ideal_pairs
            DataDict["dyn_params"] = dp
            DataDict["up_params"] = up
            DataDict["thresholds"] = thresholds
            jldsave(save_loc*"/graph_$(completed_graphs).jld2"; fp_data = DataDict)
        end

    end
    DataDict = Dict()
    DataDict["fp_list"] = fp_list
    DataDict["ideal_pairs"] = ideal_pairs_list
    DataDict["dyn_params"] = dyn_params
    DataDict["up_params"] = up_list
    
    toml_file_name = save_loc*"/"*string(today())*"_"*string(Dates.hour(now()))*string(Dates.minute(now()))*".toml"

    grn_dict = Configurations.to_dict(up)

    open(toml_file_name, "w") do io
        TOML.print(io, grn_dict)
    end

    
end

function search_for_fp(dp, stable, func;
    num_fp_to_test = 300,
    time_to_find_fp = [0 200])
    fp_tree = KDTree(reshape(stable, length(stable),1); leaf_size = 1)
    #binaries = [[stable.>=0.5]]
    du = zeros(nv(dp.G))
    fp_vec = [stable]
    thresholds = 0.5*ones(nv(dp.G))
    for i in 1:num_fp_to_test
        prob = ODEProblem(func, rand(nv(dp.G)), time_to_find_fp, dp)
        sol = solve(prob, Vern9(), save_everystep = false)
        
        if !is_fixedpoint(func, sol[end],du, dp)
            continue
        end

        test_fp = sol.u[end]
        ind, dist = nn(fp_tree, sol.u[end])

        if (dist > 1e-2)
            add_point!(fp_tree, sol.u[end])
            push!(fp_vec, sol.u[end])
        end
    end
    ideal_pairs = []
    for combo in combinations(fp_vec,2)
        activated_1 = combo[1] .>= thresholds
        activated_2 = combo[2] .>= thresholds

        if sum(activated_1 .== activated_2) == (nv(dp.G) - 1)
            push!(ideal_pairs, (combo[1], combo[2]))
        end
    end
    return fp_vec, ideal_pairs, thresholds
    
end

function make_n_graphs(N, K, ϵ, n_graphs, func; dynamics_params = nothing, force_self_loops = false)

    all_graphs = []
    all_params = []
    graphs_made = 0
    while graphs_made < n_graphs
        G,p,_ = random_gene_network(N, K, ϵ; dynamics_params = dynamics_params, force_self_loops = force_self_loops)
        du = zeros(nv(G))
        if is_connected(G)
            prob = ODEProblem(func, rand(nv(G)), [0 5000], p)
            sol = solve(prob, Vern9(), save_everystep = false, callback = TerminateSteadyState())

            if sol.t[end] == 5000
                continue
            elseif is_fixedpoint(func, sol[end], du, p)
                push!(all_graphs, G)
                push!(all_params, p)
                graphs_made += 1
            else
                continue
            end
        end
    end

    return all_graphs, all_params
end

function find_all_stable(all_graphs, all_params, func)

    all_stable = []
    for (p,G) in zip(all_params, all_graphs)
        du = zeros(nv(G))
        prob = ODEProblem(func, rand(nv(G)), [0 5000], p)
        sol = solve(prob, Vern9(), save_everystep = false, abstol = 1e-8, reltol = 1e-6, callback = TerminateSteadyState())

        if is_fixedpoint(func, sol[end], du, p)
            push!(all_stable, sol[end])
        else
            println("Error! Not a fixed point. Use a new seed.")dfcx
        end
    end

    return all_stable
end

function grn_test_parameters(func, ngp, num_graphs, seed, dynamics_params, save_loc; force_self_loops = false)
    @unpack N, K, ϵ = ngp

    if !isnothing(seed)
        Random.seed!(seed)
    end

    all_graphs, all_params = make_n_graphs(N, K, ϵ, num_graphs, func; dynamics_params = dynamics_params, force_self_loops = force_self_loops)

    all_stable = find_all_stable(all_graphs, all_params, func)

    all_fp = []
    all_fp_counts = []

    for (p, stable) in zip(all_params, all_stable)
        fps, fp_counts = search_for_fp(p, stable, func)
        push!(all_fp, fps)
        push!(all_fp_counts, fp_counts)
    end

    save_fp_data(save_loc, N, K, ϵ, all_fp_counts, all_params, dynamics_params, func)

end

function save_fp_data(save_loc, N, K, ϵ, fp_counts, all_params, dynamics_params, func)
    DataDict = Dict()
    DataDict["fp_counts"] = fp_counts
    DataDict["dyn_params"] = all_params
    mkpath(save_loc)
    jldsave(save_loc*"/N_$(N)_K_$(K)_e_$(ϵ)_dp_$(dynamics_params[1])_$(dynamics_params[2]).jld2"; data_dict = DataDict)

end

function make_grn_varying_n(save_loc, N_vec = [10, 15, 20, 25, 30, 35, 40, 45, 50];
    dyn_params = [0.5, 4],
    avg_degree = 6,
    ϵ = 0.3,
    func = grn_function,
    n_graphs = 20)

    addprocs(5)

    @everywhere include("initialize_grn.jl")
    @everywhere include("scripts/gene_regulatory/grn_generation.jl")
    @everywhere include("scripts/gene_regulatory/grn_function.jl")
    @everywhere include("scripts/indirect_targeting_algorithm/FastMTK.jl")
    @everywhere include("scripts/indirect_targeting_algorithm/perturbations.jl")


    pmap(x -> generate_grns(n_graphs, save_loc*"_$(x)_nodes", func; dynamics_params = dyn_params, toml_file_name = "GRN_NetworkGen_", force_self_loops = false, ngp_custom = [x, avg_degree, ϵ]), N_vec)
    rmprocs(workers())
end

