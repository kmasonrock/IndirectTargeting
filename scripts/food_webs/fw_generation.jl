function generate_node_params()
    η = rand(Uniform(0,1))
    r = rand(Beta(1,1.5))*η
    centre = rand(Uniform(r/2, η))
    c = (centre - r/2, centre + r/2)
    return η, r, c
end

function is_fixedpoint(func, vals, du, p, thresh)
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

function max_basal(g, basal_limit, n_nodes)
    mkpath("LinAlgFails")
    try
        tl = get_trophic(g)
        if (sum(tl .<= 1)/n_nodes < basal_limit)
            return false
        else
            return true
        end
    catch e
        return true
    end
end

function cycle_check(g)
    for n in vertices(g)
        in_nodes = [x[1] for x in sort(neighborhood_dists(g, n, nv(g); dir = :in))]
        out_nodes = [x[1] for x in sort(neighborhood_dists(g, n, nv(g); dir = :out))]

        deleteat!(in_nodes, findall(x -> x == n, in_nodes))
        deleteat!(out_nodes, findall(x -> x == n, out_nodes))

        if any(x -> x in out_nodes, in_nodes) | any(x -> x in in_nodes, out_nodes)
            println("Cycle found at Node $n")
        end
    end
end


function has_food_chain(G, s)
    pred,prey,basal = get_pred_and_prey(G)
    
    basal_species = findall(x -> x == 1, basal)

    for b in basal_species
        if has_path(G, b, s)
            return true
        end
    end
    return false
end

function all_have_food_chains(G)
    return all([has_food_chain(G, s) for s in vertices(G)])
end

function trophically_distinct(G, s1, s2)
    have_same_pred = Set(G.fadjlist[s1]) == Set(G.fadjlist[s2])
    have_same_prey = Set(G.badjlist[s1]) == Set(G.badjlist[s2])
    return !(have_same_pred & have_same_prey)
end

function all_trophically_distinct(G)
    return all([trophically_distinct(G, s1, s2) for (s1,s2) in combinations(vertices(G), 2)])
end

function test_dynamics(g, func, u0, dyn_p, net_gen_p)
    @unpack e_herb, e_carn, y, dxdr, Z, m_trophic, w, B0, h, m_migrate, dyn_abstol, dyn_reltol, allee_type = dyn_p
    @unpack β, ex_thresh, fp_thresh, abstol, reltol, integrate_time, starting_B, basal_limit, iter_limit = net_gen_p
    local p

    init_conditions = copy(u0)
    append!(init_conditions, starting_B)

    du = zeros(length(init_conditions))
    J = zeros(length(init_conditions), length(init_conditions))
    p = cr_params(g, dyn_p, allee_type)
    prob = ODEProblem(func, init_conditions, integrate_time, p)
    sol = solve(prob, Vern9(), save_everystep = false, abstol = dyn_abstol, reltol = dyn_reltol, callback = TerminateSteadyState(dyn_abstol, dyn_reltol), verbose = false)
    if all(x -> x > ex_thresh, sol[end]) & is_fixedpoint(func, sol[end], du, p, fp_thresh) & is_stable(func, sol[end], du, p)
        return true, sol[end], p
    else
        return false, u0, p
    end
end

function find_targets(g, func, u0, up, p)

    dyn = up.dyn
    net_gen = up.net_gen

    if up.log_space
        m = log(1e-9)
    else
        m = 1e-9
    end

    target_nodes = Vector{Int64}(undef, 0)
    target_states = Vector{Vector{Float64}}(undef, 0)

    du = zeros(nv(g))

    for v in vertices(g)

        local sol

            init_vals = deepcopy(u0)
            init_vals[v] = m

            prob = ODEProblem(func, init_vals, net_gen.integrate_time, p)
            sol = solve(prob, Vern9(), save_everystep = false)
            if (sol[end][v] < net_gen.ex_thresh) & all(x -> x > net_gen.ex_thresh, sol[end][Not(v)]) & is_fixedpoint(func, sol[end], du, p, net_gen.fp_thresh) & is_stable(func, sol[end], du, p)
                push!(target_nodes, v)
                push!(target_states, sol[end])
            end
    end
    return target_nodes, target_states 
end

function initiate_niche(starting_B)
    η = rand(Uniform(0,1))
    r = 0.0
    centre = rand(Uniform(r/2, η))
    c = (centre,centre)
    g = SimpleDiGraph(1)
    return [starting_B], [η], [r], [c], g
end

function get_resources(η_vec, c_vec)
    return findall(x -> (c_vec[end][1] <= x) & (c_vec[end][2] >= x), η_vec)
end

function get_consumers(η_vec, c_vec)
    return findall(x -> (x[1] <= η_vec[end]) & (x[2] >= η_vec[end]), c_vec)
end

function has_cycles(g)
    return isempty(simplecycles(g)) ? false : argmax(length.(simplecycles(g))) > 1
end

function add_species(u0, η0, r0, c0, g)
    η, r, c = generate_node_params()
end

function network_generation(n_nodes, func, universal_params; saveloc = nothing)

    local net_gen, dyn, p
    
    try
        net_gen = universal_params.net_gen
        dyn = universal_params.dyn
        opt = universal_params.opt
    catch
        throw("NetworkGeneration Parameters are not in the universal_params variable! Please make sure that you're providing the universal parameters, and ensure that you generate them using the initialize() function.")
    end

    u0, η_vec, r_vec, c_vec, g = initiate_niche(net_gen.init_B)
    iterations = 1
    then = time()
    while (nv(g) < n_nodes)

        if iterations >= net_gen.iter_limit
            u0, η_vec, r_vec, c_vec, g = initiate_niche(net_gen.init_B)
            iterations = 1
        end

        add_vertex!(g)
        η, r, c = generate_node_params()

        for (i,_n) in enumerate(η_vec)
            if (c[1] <= _n) & (c[2] >= _n)
                add_edge!(g, i, nv(g))
            end
        end

        for (i,v) in enumerate(c_vec)
            if (v[1] <= η) & (v[2] >= η)
                add_edge!(g,nv(g), i)
            end
        end

        if !is_connected(g)
            rem_vertex!(g,nv(g))
            iterations += 1
            continue
        end

        if has_cycles(g)
            rem_vertex!(g, nv(g))
            iterations += 1
            continue
        end

        if max_basal(g, net_gen.basal_limit, n_nodes)
            rem_vertex!(g, nv(g))
            iterations += 1
            continue
        end
        
        local working_du

        working_du, u0, p = test_dynamics(g, func, u0, dyn, net_gen)

        if !working_du
            rem_vertex!(g,nv(g))
            iterations += 1
            continue
        end
        append!(η_vec, η); append!(r_vec, r); push!(c_vec, c);
    end
    target_nodes, target_states = find_targets(g, func, u0, universal_params, p)
    
    universal_params.opt.targets = target_states
    universal_params.opt.invaders = Float64.(target_nodes)

    save_struct = SaveObject(universal_params, p, u0)
    if !isnothing(saveloc)
        jldsave(saveloc; network_params = save_struct)
        return nothing
    else
        return save_struct, iterations
    end
end

function rem_all_edges(G)
    possible_edges = vec(collect(Iterators.product(Iterators.repeated(vertices(G),2)...)))
    [rem_edge!(G, i, j) for (i,j) in possible_edges]
    return G
end

function gen_networks(n_nodes, n_networks, func, universal_params::UniversalParams, savefolder::String)
    
    mkpath(savefolder)

    for i in ProgressBar(1:n_networks)
        network_generation(n_nodes, func, universal_params; saveloc = savefolder*"/graph_$(i).jld2")
    end
end

function n_node_food_webs(parent_folder_starter::String, up::UniversalParams; 
    func = log_cr_f!, n_networks = 10, n_nodes = collect(10:5:50))
    
    addprocs(5)

    @everywhere include("initialize_fw.jl")
    @everywhere include("scripts/food_webs/consumer_resource.jl")
    @everywhere include("scripts/food_webs/fw_generation.jl")

    pmap(n -> gen_networks(n, n_networks, func, up, parent_folder_starter*"_$(n)"), n_nodes)
    rmprocs(workers())
end
