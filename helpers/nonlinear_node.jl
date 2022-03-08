using Optim: optimize, LBFGS
using ForwardDiff

struct NonlinearNode end # Dummy structure just to make Julia happy

struct NonlinearMeta{F}
    fn       :: F   # Nonlinear function, we assume 1 float input - 1 float ouput
    nsamples :: Int # Number of samples used in approximation
    ysprev   :: Vector{Float64} # Previous outputs
    us       :: Vector{Float64} # Controls
    seed     :: Int
end

@node NonlinearNode Deterministic [ out, in ]

# Rule for outbound message on `out` edge given inbound message on `in` edge
@rule NonlinearNode(:out, Marginalisation) (m_in::NormalDistributionsFamily, meta::NonlinearMeta) = begin
    rng = MersenneTwister(meta.seed)
    samples = [rand(rng, m_in) for _ in 1:meta.nsamples]
    def_fn(s) = meta.fn(meta.ysprev, meta.us, s)
    sl = SampleList(def_fn.(samples))
    return MvNormalMeanCovariance(mean(sl), cov(sl))
end

# Rule for outbound message on `in` edge given inbound message on `out` edge
@rule NonlinearNode(:in, Marginalisation) (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily, meta::NonlinearMeta) = begin
    def_fn(s) = meta.fn(meta.ysprev, meta.us, s)
    log_m_(x) = logpdf(m_out, def_fn(x))
    rng = MersenneTwister(meta.seed)
    samples = [rand(rng, m_in) for _ in 1:meta.nsamples]
    weights = log_m_.(samples)
    # weights = map(log_m_, samples)
    max_val = max(weights...)

    log_norm = max_val + log(sum(exp.(weights .- max_val)))

    weights = exp.(weights .- log_norm)
    # map!(w -> exp(w - log_norm), weights, weights)

    μ = sum(weights.*samples)
    # μ = mapreduce(d -> d[1] * d[2], +, zip(weights, samples))
    # tmp = similar(μ)

    tot = zeros(length(samples[1]), length(samples[1]))
    for i = 1:meta.nsamples
        # map!(-, tmp, samples[i], μ)
        # tot += tmp * transpose(tmp) .* weights[i]
        tot += (samples[i] .- μ) * transpose(samples[i] .- μ) .* weights[i]
    end
    Σ = (meta.nsamples/(meta.nsamples - 1)) .* tot
    prec = inv(Σ) - precision(m_in)
    prec_mu = inv(Σ)*μ - weightedmean(m_in)
    return MvNormalWeightedMeanPrecision(prec_mu, prec)
end

# # Laplace Approximation
# @rule NonlinearNode(:in, Marginalisation) (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily, meta::NonlinearMeta) = begin
#     def_fn(s) = meta.fn(meta.ysprev, meta.us, s)
#     log_m_(x) = logpdf(m_out, def_fn(x))

#     # Optimize with gradient ascent
#     log_joint(x) = logpdf(m_out, def_fn(x)) + logpdf(m_in, x)
#     neg_log_joint(x) = -log_joint(x)
#     d_log_joint(x) = ForwardDiff.gradient(log_joint, x)
#     m_initial = mean(m_in)

#     #mean = gradientOptimization(log_joint, d_log_joint, m_initial, 0.01)
#     μ = optimize(neg_log_joint, m_initial, LBFGS(); autodiff = :forward).minimizer
#     W = -ForwardDiff.jacobian(d_log_joint, μ)

#     prec = W - precision(m_in)
#     prec_mu = W*μ - weightedmean(m_in)
#     return MvNormalWeightedMeanPrecision(prec_mu, prec)
# end

# EKF forward
@rule NonlinearNode(:out, Marginalisation) (m_in::NormalDistributionsFamily, meta::NonlinearMeta) = begin
    def_fn(s) = meta.fn(meta.ysprev, meta.us, s)
    m_ = mean(m_in)
    P_ = cov(m_in)
    m = def_fn(m_)
    H = ForwardDiff.gradient(def_fn, m_)
    P = H*P_*transpose(H)
    return MvNormalMeanCovariance(m, P)
end

# EKF backward
@rule NonlinearNode(:in, Marginalisation) (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily, meta::NonlinearMeta) = begin
    def_fn(s) = meta.fn(meta.ysprev, meta.us, s)
    m_ = mean(m_in)
    P_ = cov(m_in)
    H = ForwardDiff.gradient(def_fn, m_)
    y = mean(m_out)
    R = cov(m_out)
    v = y - def_fn(m_)
    S = H*P_*transpose(H) + R
    K = P_*transpose(H)*inv(S)
    m = m_ + K*v
    P = P_ - K*S*transpose(K)

    return MvNormalMeanCovariance(m, P)
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
