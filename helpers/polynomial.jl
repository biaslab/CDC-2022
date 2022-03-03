using LinearAlgebra

function gen_combs(options)

    na = options["na"]
    nb = options["nb"]
    ne = options["ne"]
    nd = options["nd"]
    nk = nb+na+ne+1
    
    # Repeat powers
    comb = reshape(collect(0:nd), (1,nd+1))

    # Start combinations array
    combs = reshape(collect(0:nd), (1,nd+1))
    for _ = 2:nb+na+1

        # Current width
        width = size(combs,2)

        # Increment combinations array
        combs = [repeat(combs,1,nd+1); kron(comb,ones(1,width))]
        
        # remove combinations which have degree higher than nd
        ndComb = sum(combs,dims=1)
        combs = combs[:, vec(ndComb .<= nd)]
    end

    if options["noiseCrossTerms"]
        for _ = nb+na+2:nk

            # Current width
            width = size(combs,2)

            # Add noise cross terms
            combs = [repeat(combs,1,nd+1); kron(comb,ones(1,width))]
    
            # Keep only combinations whose total exponent is smaller than max exponent
            ndComb = sum(combs, dims=1)
            combs = combs[:, vec(ndComb .<= nd)]
        end
    else
        for ii = nb+na+2:nk
    #         noisecomb = [zeros(ii-1,1); 1]; % only linear terms
            noisecomb = [zeros(ii-1, nd); reshape(collect(1:nd), 1,nd)]
            combs = [[combs; zeros(1,size(combs,2))] noisecomb]
        end
    end

    if !options["crossTerms"]
        # Keep only columns with 1 non-zero element or dc component (all zeros)
        combs = combs[:, vec(sum(combs .> 0,dims=1) .<= 1)]
    end
    
    if !options["dc"]
        # Drop first column
        if sum(combs[:,1]) == 0
            combs = combs[:,2:end]
        else
            error("DC component was already missing.")
        end
    end

    return combs
end

function precompiled_phi(options=Dict("na"=>0, "nb"=>0, "ne"=>0, "nd"=>1, "dc"=>false, "crossTerms"=>true, "noiseCrossTerms"=>false))
    # Generate array of combinations
    return let P = gen_combs(options)
        (x) -> begin 
            # Input dimensionality
            N = length(x)

            # Check whether [na, nb, ne] matches vector dimensionality
            if size(P,1) != N; error("Generating polynomial combinations failed. Dimensionality of data vector does not match na+nb+ne+1."); end

            # Output dimensionality
            M = size(P,2)

            # Preallocate output vector
            y = zeros(eltype(x), M)

            # Iterate over combinations
            for m = 1:M

                # Temporary array
                T = zeros(eltype(x), N)

                # Iterate over elements of vector
                for n = 1:N

                    # Raise each element of x to a certain power
                    T[n] = x[n].^P[n,m]
                end

                # Add product of elements raised to powers
                y[m] = prod(T)
            end
            return y
        end
    end
end

function ϕ(x::Vector{Float64}, options=Dict("na"=>0, "nb"=>0, "ne"=>0, "nd"=>1, "dc"=>false, "crossTerms"=>true, "noiseCrossTerms"=>false))
    """
    Explanation of options:

    na = number of previous outputs, i.e., for na = 2, we include y_{k-1} and y_{k-2}.
    nb = number of previous inputs, i.e., for nb = 2, we include u_{k-1} and u_{k-2}.
    ne = number of previous noises, i.e., for ne = 2, we include e_{k-1} and e_{k-2}.
    nd = order of polynomial expansion, i.e., for nd = 3, the largest exponent is 3.
    dc = direct current, i.e., this controls whether there should be an offset (in other words, whether x^0 should be included)
    crossTerms = whether to allow terms involving mixed products between previous outputs and inputs, e.g., y_{k-1}^2 ⋅ u_{k-3}.
    noiseCrossTerms = whether to allow terms involving mixed products between data and noise, e.g., u_{k-2} ⋅ e_{k-1}^2.

    """

    # Input dimensionality
    N = length(x)
    
    # Generate array of combinations
    P = gen_combs(options)

    # Check whether [na, nb, ne] matches vector dimensionality
    if size(P,1) != N; error("Generating polynomial combinations failed. Dimensionality of data vector does not match na+nb+ne+1."); end

    # Output dimensionality
    M = size(P,2)

    # Preallocate output vector
    y = zeros(M)

    # Iterate over combinations
    for m = 1:M
        
        # Temporary array
        T = zeros(N)

        # Iterate over elements of vector
        for n = 1:N

            # Raise each element of x to a certain power
            T[n] = x[n].^P[n,m]
        end

        # Add product of elements raised to powers
        y[m] = prod(T)
    end
    return y
end