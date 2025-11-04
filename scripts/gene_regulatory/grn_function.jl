function grn_function(du, u ,p ,t)
    terms = get_tmp(p.dummy[2], u)

    inhibs = get_tmp(p.dummy[3], u)
    activs = get_tmp(p.dummy[4], u)

    @inbounds @. terms = 1/(1 + u^p.n / p.Kn)

    for (ind, a) in enumerate(p.activators)
        @views activs[ind] = foldl(*,terms[a]; init = 1)
    end

    for (ind, i) in enumerate(p.inhibitors)
        @views inhibs[ind] = foldl(*,terms[i]; init = 1)
    end

    @. du = (1 - activs)*inhibs - u
    nothing
end

    
