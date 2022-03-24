# @rule AR(:y, Marginalisation) (m_x::Any, q_θ::Any, q_γ::Any, meta::ARMeta) = begin
#     mθ, Vθ = mean_cov(q_θ)
#     mx, Wx = mean_invcov(m_x)

#     mγ = mean(q_γ)

#     mA = ReactiveMP.as_companion_matrix(mθ)
#     mV = ReactiveMP.ar_transition(ReactiveMP.getvform(meta), ReactiveMP.getorder(meta), mγ)

#     D = Wx + mγ * Vθ
#     C = mA * inv(D)

#     my = C * Wx * mx
#     Vy = ReactiveMP.add_transition!(C * mA', mV)
    
#     return convert(ReactiveMP.promote_variate_type(ReactiveMP.getvform(meta), NormalMeanVariance), my, Vy)
# end


# @rule AR(:x, Marginalisation) (m_y::Any, q_θ::NormalDistributionsFamily, q_γ::Any, meta::ARMeta) = begin
#     mθ, Vθ = mean_cov(q_θ)
#     my, Vy = mean_cov(m_y)

#     mγ = mean(q_γ)

#     mA = ReactiveMP.as_companion_matrix(mθ)
#     mV = ReactiveMP.ar_transition(getvform(meta), getorder(meta), mγ)

#     C = mA' * inv(add_transition(Vy, mV))
    
#     W = C * mA + mγ * Vθ
#     ξ = C * my

#     return convert(ReactiveMP.promote_variate_type(ReactiveMP.getvform(meta), NormalWeightedMeanPrecision), ξ, W)
# end

# @marginalrule AR(:y_x) (m_y::Any, m_x::Any, q_θ::NormalDistributionsFamily, q_γ::Any, meta::ARMeta) = begin
#     return ar_y_x_marginal(ReactiveMP.getstype(meta), m_y, m_x, q_θ, q_γ, meta)
# end

# function ar_y_x_marginal(::ARsafe, m_y::Any, m_x::Any, q_θ::NormalDistributionsFamily, q_γ::Any, meta::ARMeta)
#     mθ, Vθ = mean_cov(q_θ)
#     mγ = mean(q_γ)

#     mA = ReactiveMP.as_companion_matrix(mθ)
#     mW = ReactiveMP.ar_precision(ReactiveMP.getvform(meta), ReactiveMP.getorder(meta), mγ)

#     b_my, b_Vy = mean_cov(m_y)
#     f_mx, f_Vx = mean_cov(m_x)

#     inv_b_Vy = cholinv(b_Vy)
#     inv_f_Vx = cholinv(f_Vx)

#     D = inv_f_Vx + mγ * Vθ

#     W_11 = ReactiveMP.add_precision(inv_b_Vy, mW)

#     # Equvalent to -(mW * mA)
#     W_12 = ReactiveMP.negate_inplace!(mW * mA)

#     # Equivalent to (-mA' * mW)
#     W_21 = ReactiveMP.negate_inplace!(mA' * mW)

#     W_22 = D + mA' * mW * mA

#     W = [ W_11 W_12; W_21 W_22 ]
#     ξ = [ inv_b_Vy * b_my; inv_f_Vx * f_mx ]

#     return MvNormalWeightedMeanPrecision(ξ, W)
# end


# @rule AR(:γ, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_θ::NormalDistributionsFamily, meta::ARMeta) = begin
#     order = ReactiveMP.getorder(meta)
#     F     = ReactiveMP.getvform(meta)

#     y_x_mean, y_x_cov = mean_cov(q_y_x)
#     mθ, vθ = mean_cov(q_θ)

#     mA, Vθ = ReactiveMP.as_companion_matrix(mθ), vθ
#     my, Vy = ReactiveMP.ar_slice(F, y_x_mean, 1:order), ReactiveMP.ar_slice(F, y_x_cov, 1:order, 1:order)
#     mx, Vx = ReactiveMP.ar_slice(F, y_x_mean, (order + 1):2order), ReactiveMP.ar_slice(F, y_x_cov, (order + 1):2order, (order + 1):2order)
#     Vyx    = ReactiveMP.ar_slice(F, y_x_cov, (order + 1):2order, 1:order)

#     C = ReactiveMP.rank1update(Vx, mx)
#     R = ReactiveMP.rank1update(Vy, my)
#     L = ReactiveMP.rank1update(Vyx, mx, my)

#     B = first(R) - 2 * first(mA * L) + first(mA * C * mA') + ReactiveMP.mul_trace(Vθ, C)

#     return GammaShapeRate(convert(eltype(B), 3//2), B / 2)
# end