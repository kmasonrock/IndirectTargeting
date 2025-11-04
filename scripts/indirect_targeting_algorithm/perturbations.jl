struct FoodWebPerturb end
struct GeneRegPerturb end

#The function that calcuates the variational matrix M
function big_f(du, u, p, t)
    func = p[1]
    J = p[2]
    n = p[3]

    @views func.f(du[1:n], u[1:n], nothing, t)
    @views func.jac(J, u[1:n], nothing, t)
    @views M = reshape(u[n+1 : end], n, n)
    @views dM = reshape(du[n+1 : end], n, n)
    mul!(dM, J, M)

    @views du[n+1 : end] = reshape(dM, n, n)

end

#Checks to make sure the delta is within the defined bounds
function check_delta(delta, ϵ_min, ϵ_max)
    delta_magnitude = norm(delta)
    return (ϵ_min <= round(delta_magnitude; digits = 5)) & (round(delta_magnitude; digits = 5) <= ϵ_max)
end

#Checks to make sure the new IC is within the box
function check_new_ic(initial_value, new_value)
    return any(x -> round(x; digits = 7) .< 0, initial_value - new_value)
end

#Initialization of the optimizer function.
function opt_initialize(network_params::SaveObject)
    @unpack up, crp, stable = network_params

    n = nv(crp.g)

    func = fmtk_cr(crp; log_space = up.log_space)
    func_ic = copy(stable)
    big_ic = vcat(stable, reshape(Matrix(1.0I, n, n), n^2, 1))
    lower_bounds = zeros(n)
    lower_bounds[findall(x -> x == 1, crp.basal)] .= up.opt.basal_lb
    lower_bounds[findall(x -> x != 1, crp.basal)] .= up.opt.basal_lb

    target_states_and_nodes = zip(up.opt.targets, up.opt.invaders)

    return func, func_ic, big_ic, n, target_states_and_nodes, lower_bounds, up, crp
end


#Step 1 of the optimization: returns the time of closest approach
function opt_time_closest(func, u0, time_closest, abstol, reltol, time_inc, target_state)
    prob = ODEProblem(func, u0, time_closest)
    sol = solve(prob, Vern9(), abstol = abstol, reltol = reltol, saveat = time_inc)

    smallest_ind = argmin(get_distances(sol.u, target_state))
    return sol.t[smallest_ind]
end

#Step 2 of the optimization: Calculates the variational matrix, and returns it at
#time of closest approach 
function opt_big_function(big_func, big_u0, time_closest, big_params, abstol, reltol)
    _, _, n = big_params

    prob = ODEProblem(big_func, big_u0, time_closest, big_params)
    sol = solve(prob, Vern9(), abstol = abstol, reltol = reltol, save_everystep = false, save_start = false)

    M_tc = reshape(sol.u[1][n+1 : end], n, n)

    return sol, M_tc
end

#Step 3 of the optimization: Optimizer finds the best update to the initial condition
#that brings it closer to the target
function obj_func(dx, M_tc, target_state, big_sol, target_node, n, non_greedy)
    if non_greedy
            return sum((target_state .- (big_sol.u[1][1:n] .+ M_tc*dx)).^2)
    else
            return (target_state[target_node] .- (big_sol.u[1][1:n][target_node] .+ (M_tc*dx)[target_node]))^2
    end
end

function opt_model_perturbation(big_sol, M_tc, target_state, target_node, func_u0, ic, delta, lb, ub, opt, n, dyn_tol, non_greedy = false)
    model = Model(Ipopt.Optimizer)
    set_attribute(model, "acceptable_tol", opt.opt_abstol) #TODO PLEASE STOP DOING MAGIC NUMBERS.
    set_attribute(model, "max_iter", Int(opt.opt_maxiters))

    set_silent(model)

    JuMP.@variable(model,
    dx[i = 1:n]
    )

    obj_f = obj_func(dx, M_tc, target_state, big_sol, target_node, n, non_greedy)

    expr = JuMP.@expression(model,
    obj_f,
    )

    JuMP.@constraint(model,
    dx[target_node] == 0
    )

    JuMP.@constraint(model,
        dx .+ func_u0[1:n] .<= ub
    )

    JuMP.@constraint(model,
        dx .+ func_u0[1:n] .>= lb
    )

    JuMP.@constraint(model,
        dot(dx, delta) >= 0
    )

    JuMP.@NLconstraint(model,
        (opt.ϵ_min)^2 <= sum(dx[j]^2 for j = 1:n) <= (opt.ϵ_max)^2
    )

    JuMP.@objective(model, Min, expr)
    set_attribute(model, "hsllib", HSL_jll.libhsl_path)
    set_attribute(model, "linear_solver", "ma97")
    optimize!(model)

    if !is_solved_and_feasible(model)
        println("MODEL BROKE. FAIL DEADLY.")
        return "deadly_fail"
    end

    delta = value.(dx)

    return delta


end

#Step 4 of the optimizer: Checks to see if we made it to the target state,
#where the invasive species is dead, and the other species are alive
function opt_post_perturb_trajectory(func, u0, big_u0, δx_0, opt, dyn)

    new_u0 = u0 .+ δx_0
    new_big_u0 = copy(big_u0)
    @views new_big_u0[1:length(u0)] .= new_u0

    prob = ODEProblem(func, new_u0, opt.integrate_time)
    sol = solve(prob, Vern9(), abstol = dyn.dyn_abstol, reltol = dyn.dyn_reltol)
    return new_u0, new_big_u0, sol

end

#Used to find the distances between two trajectories
function get_distances(u,t)
    return [norm(i .- t) for i in u]
end

#The optimization function, combining all 4 steps and checks in between to ensure
#the code is providing valid solutions.
function optimization(func::ODEFunction, func_ic, big_ic, target_state, target_node::Int64, opt::OptimizationParams, dyn::DynamicsParams, lower_bounds, upper_bounds, n, network_type::GeneRegPerturb,; opt_func_type::Int64 = 0, opt_abstol = 1e-8, opt_reltol = 1e-6)

    lb = lower_bounds
	check_atol = 1e-8
    check_rtol = 1e-6
    local sol
    local greedy_fail
    local δx_0

    binary_ic = func_ic .> 0.5

	
    historical_ic = [] #selected initial conditions
	
    J = zeros(n,n)

    func_u0 = copy(func_ic)
    big_u0 = copy(big_ic)
    delta = zeros(n)
    iter = 1
    if opt_func_type == 0
        #Greedy
        is_hybrid = false
        greedy_fail = false
    elseif opt_func_type == 1
        #Non-greedy
        is_hybrid = false
        greedy_fail = true
    elseif opt_func_type == 2
        #Hybrid
        #println("hybrid type")
        is_hybrid = true
        greedy_fail = false
    else
        throw("Function type is invalid. Please provide numbers 0,1, or 2.")
        return nothing
    end
    #opt.closest_time = [0.0, 2.5]

    max_iter = opt.iter_limit

    while iter < max_iter 
        tc = opt_time_closest(func, func_u0, opt.closest_time, dyn.dyn_abstol, dyn.dyn_reltol, opt.time_increment, target_state)

        big_sol, M_tc = opt_big_function(big_f, big_u0, (0, tc), (func, J, n), dyn.dyn_abstol, dyn.dyn_reltol)

        δx_0 = opt_model_perturbation(big_sol, M_tc, target_state, target_node, func_u0, func_ic, delta, lower_bounds, upper_bounds, opt, n, opt.opt_abstol, greedy_fail)

        if typeof(δx_0) == String
            break
        end

        if norm(M_tc * δx_0) > (opt.ϵ_max)
            δx_0 = (opt.ϵ_max .* δx_0)/norm(M_tc * δx_0)
        end 

        new_u0, new_big_u0, sol = opt_post_perturb_trajectory(func, func_u0, big_u0, δx_0, opt, dyn)

        func_u0 = new_u0
        big_u0 = new_big_u0
        delta = δx_0
        iter += 1

        binary_u0 = sol[end] .> 0.5
                
        if isapprox(sol[end], target_state; atol = check_atol, rtol = check_rtol)
            println("Success")
            return sol.u[end], "Success", func_u0

        elseif (binary_u0 != binary_ic) & is_hybrid & !greedy_fail
            greedy_fail = true
            iter = 0
        end
    end

    println("Failure")
    retcode = "Failure"
    return sol.u[end], retcode, func_u0 #historical_delta, historical_ic, historical_sol, historical_tc
end

function optimization(func::ODEFunction, func_ic, big_ic, target_state, target_node::Int64, opt::OptimizationParams, dyn::DynamicsParams, lower_bounds, upper_bounds, n, network_type::FoodWebPerturb; opt_func_type::Int64 = 0, opt_abstol = 1e-8, opt_reltol = 1e-6)

    local sol
    local greedy_fail
    local δx_0
    check_atol = 1e-8
    check_rtol = 1e-6

	
    historical_ic = [] #selected initial conditions
	
    J = zeros(n,n)

    func_u0 = copy(func_ic)
    big_u0 = copy(big_ic)
    delta = zeros(n)
    iter = 1
    if opt_func_type == 0
        #Greedy
        is_hybrid = false
        greedy_fail = false
    elseif opt_func_type == 1
        #Non-greedy
        is_hybrid = false
        greedy_fail = true
    elseif opt_func_type == 2
        #Hybrid
        #println("hybrid type")
        is_hybrid = true
        greedy_fail = false
    else
        throw("Function type is invalid. Please provide numbers 0,1, or 2.")
        return nothing
    end
    #opt.closest_time = [0.0, 2.5]

    max_iter = opt.iter_limit

    while iter < max_iter 
        tc = opt_time_closest(func, func_u0, opt.closest_time, opt.opt_abstol, opt.opt_reltol, opt.time_increment, target_state)

        big_sol, M_tc = opt_big_function(big_f, big_u0, (0, tc), (func, J, n), opt.opt_abstol, opt.opt_reltol)

        δx_0 = opt_model_perturbation(big_sol, M_tc, target_state, target_node, func_u0, func_ic, delta, lower_bounds, upper_bounds, opt, n, opt.opt_abstol, greedy_fail)

        if typeof(δx_0) == String
            break
        end

        if norm(M_tc * δx_0) > (opt.ϵ_max)
            δx_0 = (opt.ϵ_max .* δx_0)/norm(M_tc * δx_0)
        end 

        new_u0, new_big_u0, sol = opt_post_perturb_trajectory(func, func_u0, big_u0, δx_0, opt, dyn)

        func_u0 = new_u0
        big_u0 = new_big_u0
        delta = δx_0
        iter += 1

        push!(historical_ic, func_u0)
        if isapprox(sol[end], target_state; atol = check_atol, rtol = check_rtol)
            
            println("Success")
            return sol.u[end], "Success", func_u0
        elseif !isapprox(sol[end][target_node], target_state[target_node]; atol = check_atol, rtol = check_rtol) & is_hybrid & !greedy_fail
            
            greedy_fail = true
            iter = 0
        end

    end
    println("Failure")
    retcode = "Failure"
    return sol.u[end], retcode, func_u0 #historical_delta, historical_ic, historical_sol, historical_tc
end

#Automates the optimization algorithm by loading each file from the given folder, and doing
#perturbations on all graphs within.
function perturb_function(file_loc::String, save_loc::String, func, network_type::FoodWebPerturb; opt_func_type = 2)

    files = glob(file_loc*"/*.jld2")
    
    if !isdir(save_loc)
        mkpath(save_loc)
    end

    for file in files
        f = jldopen(file, "r")

        target_states = f["network_params"].up.opt.targets
        target_nodes = f["network_params"].up.opt.invaders
        p = f["network_params"].crp
        n = nv(p.g)
        fmtk_func = fmtk_initialize_model(func, p, n, nothing)
        func_ic = copy(f["network_params"].stable)
        big_ic = vcat(func_ic, reshape(Matrix(1.0I, n, n), n^2, 1))
        lower_bounds = ones(n).*f["network_params"].up.opt.basal_lb
        upper_bounds = func_ic

        for (target_state, target_node) in zip(target_states, target_nodes)
            #, historical_delta, historical_ic, historical_sol, historical_tc
            sol, retcode, new_u0 = optimization(fmtk_func, func_ic, big_ic, target_state, Int(target_node), f["network_params"].up.opt, f["network_params"].up.dyn, lower_bounds, upper_bounds, n, network_type; opt_func_type = opt_func_type) #2 for hybrid approach
            #, deltas = historical_delta, ics = historical_ic, sols = historical_sol, historical_tc = historical_tc, up = up, crp = crp
            jldsave(save_loc*split(file,file_loc)[end][1:end-5]*"_$(Int(target_node)).jld2"; end_state = sol, retcode = retcode, perturbation = new_u0, used_params = f["network_params"])
        end
        close(f)
    end

    nothing
end

function perturb_function(file_loc::String, save_loc::String, func, network_type::GeneRegPerturb; opt_func_type = 2)
    files = glob(file_loc*"/graph_*.jld2")
    if !isdir(save_loc)
        mkpath(save_loc)
    end

    for file in files
        f = jldopen(file, "r")
        n = nv(f["fp_data"]["dyn_params"].G)
        lower_bounds = zeros(n)
        upper_bounds = ones(n)
        println(file)
        fmtk_func = fmtk_initialize_model(func, f["fp_data"]["dyn_params"], nv(f["fp_data"]["dyn_params"].G), nothing)
        ideal_pairs = f["fp_data"]["ideal_pairs"]
        thresholds = f["fp_data"]["thresholds"]

        for (ind,pair) in enumerate(ideal_pairs)

            for j in 1:2
                @views stable = pair[j]
                @views target_state = pair[2 - (j-1)]

                stable_on = stable .> thresholds
                target_on = target_state .> thresholds

                target_node = findall(x -> abs(x) == 1, stable_on - target_on)[1]

                func_ic = copy(stable)
                big_ic = vcat(func_ic, reshape(Matrix(1.0I, n, n), n^2, 1))
                sol, retcode, new_u0 = optimization(fmtk_func, func_ic, big_ic, target_state, Int(target_node[1]), f["fp_data"]["up_params"].opt, f["fp_data"]["dyn_params"], lower_bounds, upper_bounds, n, network_type; opt_func_type = opt_func_type)
                #, deltas = historical_delta, ics = historical_ic, sols = historical_sol, historical_tc = historical_tc, up = up, crp = crp
                jldsave(save_loc*split(file,file_loc)[end][1:end-5]*"_$(Int(target_node[1]))_$(ind)_$(j).jld2"; end_state = sol, retcode = retcode, perturbation = new_u0)
            end
        end    
        close(f)
        GC.gc()
    end
    println("$file_loc complete.")

    nothing
end

function perturb_FWs(parents::Vector{String}, save_ender::String, network_type = FoodWebPerturb())

    save_locs = parents.*save_ender

    addprocs(5)

    @everywhere include("initialize_fw.jl")
    @everywhere include("scripts/food_webs/consumer_resource.jl")
    @everywhere include("scripts/indirect_targeting_algorithm/FastMTK.jl")
    @everywhere include("scripts/indirect_targeting_algorithm/perturbations.jl")


    pmap(i -> perturb_function(parents[i], save_locs[i], log_cr_f!, network_type; opt_func_type = 2), eachindex(parents))

    rmprocs(workers())

end

function perturb_GRNs(folders::Vector{String}, save_ender::String, network_type = GeneRegPerturb())

 
    save_locs = folders.*save_ender


    addprocs(5)
    @everywhere include("initialize_grn.jl")    
    @everywhere include("scripts/gene_regulatory/grn_function.jl")
    @everywhere include("scripts/indirect_targeting_algorithm/perturbations.jl")
    @everywhere include("scripts/indirect_targeting_algorithm/FastMTK.jl")

    pmap(x -> perturb_function(folders[x], save_locs[x], grn_function, network_type; opt_func_type = 2), eachindex(folders))

	rmprocs(workers())

end





