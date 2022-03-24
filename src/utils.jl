export transition, shift, prediction, inference_callback, narmax_meta

@meta function narmax_meta(artype, order_1, order_2, stype)
    AR(h, θ, w) -> ARMeta(artype, order_1, stype)
    AR(ẑ, η, τ) -> ARMeta(artype, order_2, stype)
end

@marginalrule NonlinearNode(:in) (m_out::MvNormalWeightedMeanPrecision, m_in::MvNormalWeightedMeanPrecision, meta::NonlinearMeta) = begin
    m_in_ = @call_rule NonlinearNode(:in, Marginalisation) (m_out=m_out, m_in=m_in, meta=meta)
    return prod(ProdAnalytical(), m_in, m_in_)
end


# HELPER
struct DummyDistribution
end

Distributions.entropy(dist::DummyDistribution) = ReactiveMP.InfCountingReal(0.0, -1)

@marginalrule typeof(+)(:in1_in2) (m_out::PointMass, m_in1::NormalDistributionsFamily, m_in2::NormalDistributionsFamily, ) = begin
    return DummyDistribution()
end


function transition(γ, order)
    V = zeros(order, order)
#     V = diageye(order)
    V[1] = 1/γ
    return V
end

function shift(dim)
    S = Matrix{Float64}(I, dim, dim)
    for i in dim:-1:2
           S[i,:] = S[i-1, :]
    end
    S[1, :] = zeros(dim)
    return S
end


function prediction(h_prior, w_mle, η_posterior, τ_posterior, full_order, order_h; meta)
    h_out = @call_rule MvNormalMeanCovariance(:out, Marginalisation) (m_μ=MvNormalMeanPrecision(mean(h_prior), precision(h_prior)), q_Σ=PointMass(transition(mean(w_mle), order_h)))
    ϕ_out = @call_rule NonlinearNode(:out, Marginalisation) (m_in=h_out, meta=meta)
    ar_out = @call_rule AR(:y, Marginalisation) (m_x=ϕ_out, q_θ=η_posterior, q_γ=τ_posterior, meta=ARMeta(Multivariate, full_order, ARsafe()))
    c = zeros(full_order); c[1] = 1.0
    dot_out = @call_rule typeof(dot)(:out, Marginalisation) (m_in1=PointMass(c), m_in2=ar_out, meta=ReactiveMP.TinyCorrection())
    c = zeros(order_h); c[1] = 1.0
    c_out = @call_rule typeof(dot)(:out, Marginalisation) (m_in1=PointMass(c), m_in2=h_out, meta=ReactiveMP.TinyCorrection())
    @call_rule typeof(+)(:out, Marginalisation) (m_in1=dot_out, m_in2=c_out)    
end

function inference_callback(h_prior, η_prior, τ_prior, w_prior, Y, X, U, order_h, full_order, ϕ)
    
    narmax_imarginals = (h = h_prior,
                         w = w_prior,
                         θ = MvNormalMeanPrecision(zeros(order_h), 1e12*diageye(order_h)),
                         τ = τ_prior,
                         η = η_prior);
    
    narmax_imessages = (h = MvNormalMeanPrecision(zeros(order_h), diageye(order_h)), );
    
    narmax_model = Model(narmax, length(Y), 
                    (mean(h_prior), precision(h_prior)), 
                    (shape(w_prior), rate(w_prior)), 
                    (mean(η_prior), precision(η_prior)), 
                    (shape(τ_prior), rate(τ_prior)), 
                    X, U, order_h, full_order, ϕ);
    
    narmax_imarginals = (h = h_prior,
                         w = w_prior,
                         θ = MvNormalMeanPrecision(zeros(order_h), 1e12*diageye(order_h)),
                         τ = τ_prior,
                         η = η_prior);

    narmax_constraints = @constraints begin
        q(θ) :: Marginal(MvNormalMeanPrecision(zeros(order_h), 1e12*diageye(order_h)))
        q(ẑ, z, η, τ, h_0, h, θ, w) = q(ẑ, z)q(η)q(τ)q(h_0, h)q(θ)q(w)
    end
    
    res = inference(
                        model = narmax_model, 
                        data  = (y = Y, ),
                        constraints   = narmax_constraints,
                        meta          = narmax_meta(Multivariate, order_h, full_order, ARsafe()),
                        options       = model_options(limit_stack_depth = 500),
                        initmarginals = narmax_imarginals,
                        initmessages  = narmax_imessages,
                        returnvars    = (θ = KeepLast(), w=KeepLast(), h=KeepLast(), η=KeepLast(), τ=KeepLast(), z=KeepLast(), ẑ=KeepLast()),
                        free_energy   = true,
                        iterations    = 10, 
                        showprogress  = true
                    );

    
    @unpack θ, w, h, η, τ, z, ẑ = res.posteriors
    θ.data, w.data, h[end].data, η.data, τ.data
end