export generate_coefficients, generate_data, ssm

using Distributions
using LinearAlgebra
using Random
import PolynomialRoots.roots

function generate_coefficients(seed, order::Int)
    rng = MersenneTwister(seed)

    stable = false
    true_a = []
    # Keep generating coefficients until we come across a set of coefficients
    # that correspond to stable poles
    while !stable
        true_a = randn(rng, order)
        coefs =  append!([1.0], -true_a)
        if false in ([abs(root) for root in roots(coefs)] .> 1)
            continue
        else
            stable = true
        end
    end
    return true_a
end

function ssm(series, order)
    inputs = [reverse!(series[1:order])]
    outputs = [series[order + 1]]
    for x in series[order+2:end]
        push!(inputs, vcat(outputs[end], inputs[end])[1:end-1])
        push!(outputs, x)
    end
    return inputs, outputs
end

function generate_data(seed, ϕ, options, T=8000; 
                       N=2^16, P=1, M=1, fMin=0, fMax=1000, fs=1000, type_signal="odd",
                       uStd=0.1, w_true=2e4, scale_coef=0.5)

    rng = MersenneTwister(seed)
    delay_y = options["na"]
    delay_u = options["nb"]
    delay_e = options["ne"]

    full_order = delay_e + delay_y + delay_u
    true_order = length(ϕ(zeros(full_order+1), options))

    # Lines selection - select which frequencies to excite
    f0 = fs/N
    linesMin = Int64(ceil(fMin / f0) + 1)
    linesMax = Int64(floor(fMax / f0) + 1)
    lines = linesMin:linesMax

    # Remove DC component
    if lines[1] == 1; lines = lines[2:end]; end

    if type_signal == "full"
        # do nothing
    elseif type_signal == "odd"

        # remove even lines - odd indices
        if Bool(mod(lines[1],2)) # lines(1) is odd
            lines = lines[2:2:end]
        else
            lines = lines[1:2:end]
        end

    elseif type_signal == "oddrandom"

        # remove even lines - odd indices
        if Bool(mod(lines[1],2)) # lines(1) is odd
            lines = lines[2:2:end]
        else
            lines = lines[1:2:end]
        end

        # remove 1 out of nGroup lines
        nLines = length(lines)
        nRemove = floor(nLines / nGroup)
        removeInd = rand(rng, 1:nGroup, [1 nRemove])
        removeInd = removeInd + nGroup*[0:nRemove-1]
        lines = lines(!removeInd)
    end
    nLines = length(lines)

    # multisine generation - frequency domain implementation
    Fu = zeros(ComplexF64, N,M)

    # excite the selected frequencies
    Fu[lines,:] = exp.(2im*pi*rand(rng, nLines,M))

    # go to time domain
    u = real(ifft(Fu))

    # rescale to obtain desired rms std
    u = uStd * u ./ std(u[:,1])

    # generate P periods
    syn_input = repeat(u, outer=(P,1))[1:T];

    # Generate a noise sequence
    syn_noise = sqrt(inv(w_true))*randn(rng, T);

    # Define parameters
    η_true = scale_coef*generate_coefficients(seed, true_order)

    # Generate output signal
    syn_output = zeros(T)

    maxdelay = maximum([delay_y, delay_u, delay_e]) + 1
    for k = maxdelay:T
        # Fill delay vector
        z_k = [syn_input[k:-1:k-delay_u]; syn_output[k-1:-1:k-delay_y]; syn_noise[k-1:-1:k-delay_e]]
        # Generate output according to NARMAX model
        syn_output[k] = η_true'*ϕ(z_k, options) + syn_noise[k]
    end

    syn_input, syn_noise, syn_output, η_true
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