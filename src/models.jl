export narmax, narmax_mini

# Nonlinear MAX model definition
# @model function narmax(n, h_prior, w_prior, η_prior, τ_prior, y_prev, u, h_order, full_order, phi)
#     obs_prec = 1e12  # softening plus
    
#     # initialize variables
#     θ  = randomvar()
#     w  = randomvar()
#     h  = randomvar(n)
#     z  = randomvar(n)
#     ẑ  = randomvar(n)
#     y  = datavar(Float64, n)
    
#     h_0 ~ MvNormalMeanPrecision(h_prior[1], h_prior[2]) where {q=MeanField()}
#     w   ~ GammaShapeRate(w_prior[1], w_prior[2])
#     θ   ~ MvNormalMeanPrecision(zeros(h_order), diageye(h_order))
    
#     η   ~ MvNormalMeanPrecision(η_prior[1], η_prior[2])
#     τ   ~ GammaShapeRate(τ_prior[1], τ_prior[2])
    
#     b = zeros(full_order); b[1] = 1.0;
#     c = zeros(h_order); c[1] = 1.0;
    
#     h_prev = h_0
#     for t in 1:n
#         h[t] ~ AR(h_prev, θ, w)
#         z[t] ~ NonlinearNode(h[t]) where {pipeline=RequireInbound(in=MvNormalMeanPrecision(zeros(h_order), diageye(h_order))), meta = NonlinearMeta(phi, y_prev[t], u[t], 42)}
#         ẑ[t] ~ AR(z[t], η, τ)
        
#         y[t] ~ dot(ẑ[t], b) + dot(h[t], c)
        
#         h_prev = h[t]
#     end

#     return θ, w, h, η, τ, z, ẑ, y
# end

# Nonlinear MAX model definition
@model function narmax(n, h_prior, w_prior, η_prior, τ_prior, y_prev, u, delay_e, order, ϕ, approximation)
        
    S = shift(delay_e); c = zeros(delay_e); c[1] = 1.0;
    
    # initialize variables
    h  = randomvar(n-1)
    e  = randomvar(n)
    z  = randomvar(n)
    ẑ  = randomvar(n)
    y  = datavar(Float64, n)
        
    # priors
    w  ~ GammaShapeRate(shape(w_prior), rate(w_prior))
    η  ~ MvNormalMeanPrecision(mean(η_prior), precision(η_prior))
    τ  ~ GammaShapeRate(shape(τ_prior), rate(τ_prior))
    
    # initial
    h_0  ~ MvNormalMeanPrecision(mean(h_prior), precision(h_prior))
    z[1] ~ NonlinearNode(h_0) where {pipeline=RequireInbound(in=MvNormalMeanPrecision(zeros(delay_e), diageye(delay_e))), meta = NonlinearMeta(approximation, ϕ, y_prev[1], u[1])}
    ẑ[1] ~ AR(z[1], η, τ)
    
    b = zeros(order); b[1] = 1.0;
    
    h_prev = h_0
    for t in 1:n-1
        
        e[t] ~ NormalMeanPrecision(0.0, w)
        h[t] ~ S*h_prev + c*e[t]
        y[t] ~ dot(ẑ[t], b) + e[t]
    
        h_prev = h[t]
        z[t+1] ~ NonlinearNode(h_prev) where {pipeline=RequireInbound(in=MvNormalMeanPrecision(zeros(delay_e), diageye(delay_e))), meta = NonlinearMeta(approximation, ϕ, y_prev[t+1], u[t+1])}
        ẑ[t+1] ~ AR(z[t+1], η, τ)
    end
    
    e[n] ~ NormalMeanPrecision(0.0, w)
    y[n] ~ dot(ẑ[n], b) + e[n]

    return w, h, η, τ, z, ẑ, y
end

# Nonlinear MAX model definition
# @model function narmax_mini(h_prior, w_prior, η_prior, τ_prior, y_prev, u, delay_e, order)
        
#     S = shift(delay_e); c = zeros(delay_e); c[1] = 1.0;
#     h = randomvar()
#     # initialize variables
#     y  = datavar(Float64)
#     # priors
#     w  ~ GammaShapeRate(shape(w_prior), rate(w_prior))
#     η  ~ MvNormalMeanPrecision(mean(η_prior), precision(η_prior))
#     τ  ~ GammaShapeRate(shape(τ_prior), rate(τ_prior))
#     # initial
#     h_0  ~ MvNormalMeanPrecision(mean(h_prior), precision(h_prior))
#     z ~ NonlinearNode(h_0) where {pipeline=RequireInbound(in=MvNormalMeanPrecision(zeros(delay_e), diageye(delay_e))), meta = NonlinearMeta(ET(), phi_, y_prev, u)}
#     ẑ ~ AR(z, η, τ)
#     e ~ NormalMeanPrecision(0.0, w)
#     b = zeros(order); b[1] = 1.0;
#     y ~ dot(b, ẑ) + e
#     h ~ S*h_0 + c*e
#     h ~ MvNormalMeanPrecision(zeros(delay_e), diageye(delay_e))

#     return
# end

@model function narmax_mini(h_prior, w_prior, η_prior, τ_prior, y_prev, u, delay_e, order, ϕ, approximation)
    S = shift(delay_e); c = zeros(delay_e); c[1] = 1.0;
    h = randomvar()
        
    # initialize variables
    y  = datavar(Float64)
    # priors
    w  ~ GammaShapeRate(shape(w_prior), rate(w_prior))
    η  ~ MvNormalMeanPrecision(mean(η_prior), precision(η_prior))
    τ  ~ GammaShapeRate(shape(τ_prior), rate(τ_prior))
    # initial
    h_0  ~ MvNormalMeanPrecision(mean(h_prior), precision(h_prior))
    z ~ NonlinearNode(h_0) where {pipeline=RequireInbound(in=MvNormalMeanPrecision(zeros(delay_e), diageye(delay_e))), meta = NonlinearMeta(approximation, ϕ, y_prev, u)}
    ẑ ~ AR(z, η, τ)
    e ~ NormalMeanPrecision(0.0, w)
    b = zeros(order); b[1] = 1.0;
    y ~ dot(b, ẑ) + e

    h ~ S*h_0 + c*e
    h ~ MvNormalMeanPrecision(zeros(delay_e), diageye(delay_e))

    return
end

# @meta function narmax_meta(artype, order_1, order_2, stype)
#     AR(h, θ, w) -> ARMeta(artype, order_1, stype)
#     AR(ẑ, η, τ) -> ARMeta(artype, order_2, stype)
# end