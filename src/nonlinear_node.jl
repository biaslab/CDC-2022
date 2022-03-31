export NonlinearNode, NonlinearMeta, ET, UT

struct NonlinearNode end # Dummy structure just to make Julia happy

struct ET end
struct UT end

struct NonlinearMeta{T, F}
    type     :: T
    fn       :: F   # Nonlinear function, we assume 1 float input - 1 float ouput
    ysprev   :: Vector{Float64} # Previous outputs
    us       :: Vector{Float64} # Controls
end

@node NonlinearNode Deterministic [ out, in ]


# EKF forward
@rule NonlinearNode(:out, Marginalisation) (m_in::NormalDistributionsFamily, meta::NonlinearMeta{ET}) = begin
    @warn "Method is broken"
    def_fn(s) = meta.fn(meta.us, meta.ysprev, s)
    m_ = mean(m_in)
    P_ = cov(m_in)
    m = def_fn(m_)
    H = ForwardDiff.jacobian(def_fn, m_)
    P = H*P_*transpose(H)
    return MvNormalMeanCovariance(m, P)
end

# EKF backward
@rule NonlinearNode(:in, Marginalisation) (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily, meta::NonlinearMeta{ET}) = begin
    @warn "Method is broken"
    def_fn(s) = meta.fn(meta.us, meta.ysprev, s)
    m_ = mean(m_in)
    P_ = cov(m_in)
    H = ForwardDiff.jacobian(def_fn, m_)
    y = mean(m_out)
    R = cov(m_out)
    v = y - def_fn(m_)
    S = H*P_*transpose(H) + R
    K = P_*transpose(H)*inv(S)
    m = m_ + K*v
    P = P_ - K*S*transpose(K)

    prec = inv(P) - precision(m_in)
    prec_mu = inv(P)*m - weightedmean(m_in)
    return MvNormalWeightedMeanPrecision(prec_mu, prec)
end

@marginalrule NonlinearNode(:in) (m_out::MultivariateNormalDistributionsFamily, m_in::MultivariateNormalDistributionsFamily, meta::NonlinearMeta) = begin
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


## Non-linear node rules UKF 

const default_alpha = 1e-3 # Default value for the spread parameter
const default_beta = 2.0
const default_kappa = 0.0

## Return the sigma points and weights for a Gaussian distribution

function sigmaPointsAndWeights(m::Vector{Float64}, V::AbstractMatrix; alpha=default_alpha, beta=default_beta, kappa=default_kappa)
    d = length(m)
    lambda = (d + kappa)*alpha^2 - d

    sigma_points = Vector{Vector{Float64}}(undef, 2*d+1)
    weights_m = Vector{Float64}(undef, 2*d+1)
    weights_c = Vector{Float64}(undef, 2*d+1)

    if isa(V, Diagonal)
        L = sqrt((d + lambda)*V) # Matrix square root
    else
        L = sqrt(Hermitian((d + lambda)*V))
    end

    sigma_points[1] = m
    weights_m[1] = lambda/(d + lambda)
    weights_c[1] = weights_m[1] + (1 - alpha^2 + beta)
    for i = 1:d
        sigma_points[2*i] = m + L[:,i]
        sigma_points[2*i+1] = m - L[:,i]
    end
    weights_m[2:end] .= 1/(2*(d + lambda))
    weights_c[2:end] .= 1/(2*(d + lambda))

    return (sigma_points, weights_m, weights_c)
end

function unscentedStatistics(m::Vector{Float64}, V::AbstractMatrix, g::Function; alpha=default_alpha, beta=default_beta, kappa=default_kappa)
    (sigma_points, weights_m, weights_c) = sigmaPointsAndWeights(m, V; alpha=alpha, beta=beta, kappa=kappa)
    d = length(m)

    g_sigma = g.(sigma_points)
    m_tilde = sum([weights_m[k+1]*g_sigma[k+1] for k=0:2*d])
    V_tilde = sum([weights_c[k+1]*(g_sigma[k+1] - m_tilde)*(g_sigma[k+1] - m_tilde)' for k=0:2*d])
    C_tilde = sum([weights_c[k+1]*(sigma_points[k+1] - m)*(g_sigma[k+1] - m_tilde)' for k=0:2*d])

    return (m_tilde, V_tilde, C_tilde)
end

function smoothRTSMessage(m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out)
    C_tilde_inv = pinv(C_tilde)
    V_bw_in = V_fw_in*C_tilde_inv'*(V_tilde + V_bw_out)*C_tilde_inv*V_fw_in - V_fw_in
    m_bw_in = m_fw_in + V_fw_in*C_tilde_inv'*(m_bw_out - m_tilde)

    return (m_bw_in, V_bw_in) # Statistics for backward message on in
end

"""
RTS smoother update for inbound marginal; based on (Petersen et al. 2018; On Approximate Nonlinear Gaussian Message Passing on Factor Graphs)
"""
function smoothRTS(m_tilde, V_tilde, C_tilde, m_fw_in, V_fw_in, m_bw_out, V_bw_out)
    P = cholinv(V_tilde + V_bw_out)
    W_tilde = cholinv(V_tilde)
    D_tilde = C_tilde*W_tilde
    V_in = V_fw_in + D_tilde*(V_bw_out*P*C_tilde' - C_tilde')
    m_out = V_tilde*P*m_bw_out + V_bw_out*P*m_tilde
    m_in = m_fw_in + D_tilde*(m_out - m_tilde)

    return (m_in, V_in) # Statistics for marginal on in
end


@rule NonlinearNode(:out, Marginalisation) (m_in::NormalDistributionsFamily, meta::NonlinearMeta{UT}) = begin
    s_len = length(mean(m_in))
    u_len = length(meta.us)
    y_len = length(meta.ysprev)
    (m_fw_in1, V_fw_in1) = mean_cov(m_in)
    m_xys = [meta.us; meta.ysprev; m_fw_in1]
    
    def_fn(x) = meta.fn(x[1:u_len], x[u_len+1:u_len+y_len], x[u_len+y_len+1:u_len+y_len+s_len]) 
    Vxys = diageye(length(m_xys))*1e-4
        
    Vxys[end-s_len+1:end,end-s_len+1:end] = V_fw_in1

    (m_tilde, V_tilde, _) = unscentedStatistics(m_xys, Vxys, def_fn; alpha=default_alpha)
    return MvNormalMeanCovariance(m_tilde,V_tilde)
end


@rule NonlinearNode(:in, Marginalisation) (m_out::NormalDistributionsFamily, m_in::NormalDistributionsFamily, meta::NonlinearMeta{UT}) = begin
    def_fn(s) = meta.fn(meta.us, meta.ysprev, s)
    (m_fw_in1, V_fw_in1) = mean_cov(m_in)
    (m_tilde, V_tilde, C_tilde) = unscentedStatistics(m_fw_in1, V_fw_in1, def_fn; alpha=default_alpha)

    # RTS smoother
    (m_bw_out, V_bw_out) = mean_cov(m_out)
    (m_bw_in1, V_bw_in1) = smoothRTSMessage(m_tilde, V_tilde, C_tilde, m_fw_in1, V_fw_in1, m_bw_out, V_bw_out)
    return MvNormalMeanCovariance(m_bw_in1, V_bw_in1)
end