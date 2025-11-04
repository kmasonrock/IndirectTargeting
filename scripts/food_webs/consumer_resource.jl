
function _v(u,p; simplify = false) # Denominator for the F_ij function

    if simplify #Used for 
        w = 0
    else
        w = p.w
    end

    v = get_tmp(p.dummy_vals[3], u)
    for i in 1:p.n
        sum_val = 0
        for j in p.prey[i]
            sum_val += p.Ω[i,j]*u[j]^p.h
        end
        v[i] = sum_val
    end
    @. v += (p.B0^p.h) + (w *(p.B0.^p.h)*u)
    return v
end

function allee(u, a::AlleeEffect)
    return u/(a.S + u)
end

function F_ij(g,Ω,ω,h,B0,prey,B,i,j) #Not really used here per se -- but very useful to have available
    return (Ω[i,j]*(B[j]^h))/(B0^h + ω*B[i]*(B0^h) + sum(Ω[i,k]*(B[k]^ h) for k in prey; init = 0))
end

function cr_f!(du,u,p,t) #All experiments
    @unpack g,Ω,e,x,w,h,y,B0,n,m, pred, prey, basal, allee_effect, dummy_vals = p
    M = get_tmp(dummy_vals[1], u)
    X = get_tmp(dummy_vals[2], u)

    @inbounds @. M = (u - m)
    @inbounds @. X = x*y

    V = _v(u,p)
    
    for i in vertices(g)
        prey_term = 0#zero(eltype(du))
        pred_term = 0#zero(eltype(du))

        if basal[i]
            du[i] = M[i]*(1-u[i])*allee(u[i], p.allee_effect)
        else
            du[i] = -x[i]*M[i]
        end

        @inbounds @simd for j in prey[i]
            prey_term += Ω[i,j]*u[j]^h
        end

        prey_term *= X[i]*M[i]*allee(u[i],p.allee_effect)/V[i]

        @inbounds @simd for j in pred[i]
            pred_term += Ω[j,i]*u[j]*X[j]/(e[j,i] * V[j])
        end

        pred_term *= -u[i]^(h-1)*M[i]
        du[i] += prey_term + pred_term
    end
    nothing
end

function log_cr_f!(du, u , p, t)
    e_u0 = get_tmp(p.dummy_vals[4], u)
    @. e_u0 = exp(u)
    cr_f!(du, e_u0, p, t)
    @. du *= exp(-u)
    nothing
end