using DrWatson
@quickactivate :NARMAXExperiments

using Distributions
using GraphPPL
using ReactiveMP
using Random
using Plots
using LinearAlgebra
using StatsBase
import ProgressMeter

pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}");

# We create a bunch of experiments with `dict_list` function from DrWatson.jl package
# This creates a list of parameters for each experiment, we use `@onlyif` to restrict some parameters being dependant on others
experiments = dict_list(Dict(
    "type"       => "narmax",
    "n_train"    => 1000,
    "n_test"     => 1000,
    "delay"      => 3,
    "poly_order" => 2,
    "iterations" => 10,
    "w_true"     => [1e1, 1e2, 1e3, 1e4],
    "approximation" => [UT(), ET()],
    "seed"       => collect(1:10)
))

function run_experiment(experiment_params)

    # We unpack all provided parameters into separate variables
    @unpack n_train, n_test, delay, poly_order, iterations, w_true, seed, approximation = experiment_params

    delay_u, delay_y, delay_e = fill(delay, 3)
    order_u = delay_u + 1 # u_k, u_{k-1}, u_{k-2}
    
    options = Dict("na"=>delay_y, "nb"=>delay_u, "ne"=>delay_e, "nd"=>poly_order, "dc"=>true, "crossTerms"=>true, "noiseCrossTerms"=>false)

    function phi(options)
        
        precompiled = precompiled_phi(options)
        
        return (u, y, h) -> begin
            na = length(y)
            nb = length(u)-1
            ne = length(h)-1
            precompiled([u; y; h[1:end]])
        end 
    end


    
    full_order = length(ϕ(zeros(delay_y+delay_e+order_u), options))

    syn_input, syn_noise, syn_output, η_true = generate_data(seed, ϕ, options, w_true=w_true, scale_coef=0.2);
    
    train_size = n_train+order_u
    test_size = n_test+order_u

    u_train = syn_input[1:train_size]
    u_val = syn_input[train_size + 1:train_size + test_size]
    y_train = syn_output[1:train_size]
    y_val = syn_output[train_size + 1:train_size + test_size]

    # normalization
    m_y, s_y = mean(y_train), std(y_train)
    m_u, s_u = mean(u_train), std(u_train)
    output_trn = (y_train .- m_y) ./ s_y
    output_val = (y_val .- m_y) ./ s_y
    input_trn = (u_train .- m_u) ./ s_u
    input_val = (u_val .- m_u) ./ s_u;

    # Generate training data
    observations_prev, observations = ssm(output_trn, delay_y)
    controls = ssm(input_trn, order_u)[1]
    X_train, Y_train, U_train = observations_prev[1:train_size-order_u], observations[1:train_size-order_u], controls[1:train_size-order_u];

    # Generate validation data
    observations_prev, observations = ssm(output_val, delay_y)
    controls = ssm(input_val, order_u)[1]
    X_test, Y_test, U_test = observations_prev[1:test_size-order_u], observations[1:test_size-order_u], controls[1:test_size-order_u];

    h_prior, w_prior = MvNormalMeanPrecision(zeros(delay_e), diageye(delay_e)), GammaShapeRate(w_true, 1.0)
    η_prior, τ_prior = MvNormalMeanPrecision(zeros(full_order), 1e-1diageye(full_order)),  GammaShapeRate(1e2, 1.0)
    
    narmax_imessages = (e = NormalMeanPrecision(0.0, 1e-10), );

    narmax_imarginals = (h = h_prior,
                         w = w_prior,
                         τ = τ_prior,
                         η = η_prior);

    narmax_constraints = @constraints begin
        q(ẑ, z, η, τ, e, w, h, h_0) = q(ẑ, z, h, h_0)q(η)q(τ)q(e)q(w)
    end;

    ϕ_ = phi(options)
    narmax_model = Model(narmax, length(Y_train), 
                         h_prior, w_prior, η_prior, τ_prior, 
                         X_train, U_train, delay_e, full_order, ϕ_, approximation);
    



    println("Inference started")   
   
    # First execution is slow due to Julia's init compilation 
    result_inf = inference(
        model = narmax_model, 
        data  = (y = Y_train, ),
        constraints   = narmax_constraints,
        meta          = narmax_meta(Multivariate, full_order, ARsafe()),
        options       = model_options(limit_stack_depth = 500),
        initmarginals = narmax_imarginals,
        initmessages  = narmax_imessages,
        returnvars    = (w=KeepLast(), h=KeepLast(), η=KeepLast(), τ=KeepLast(), z=KeepLast(), ẑ=KeepLast()),
        free_energy   = true,
        iterations    = 100, 
        showprogress  = true
    );

    @unpack w, h, η, τ, z, ẑ = result_inf.posteriors
    println("Prediction started")              
    # prediction
    predictions = []
    h_prior = h[end].data
    w_prior = w.data
    τ_prior = τ.data
    η_prior = η.data
    
    ProgressMeter.@showprogress for i in 1:length(Y_test)
        pred = prediction(h_prior, mean(w_prior), η_prior, τ_prior, X_test[i], U_test[i], full_order=full_order, meta=NonlinearMeta(approximation, ϕ_, X_test[i], U_test[i]))
        push!(predictions, pred)
        w_post, e_post, η_post, τ_post = inference_callback(h_prior, η_prior, τ_prior, w_prior, Y_test[i], X_test[i], U_test[i], delay_e, full_order, ϕ_, approximation)
        h_prior = MvNormalMeanCovariance([mean(e_post); mean(h_prior)[2:end]], Diagonal([cov(e_post); var(h_prior)[2:end]]))
        η_prior = η_post
        τ_prior = τ_post
        w_prior = w_post
    end

    RMSE_pred = sqrt(mean((mean.(predictions) .- Y_test) .^2))

    
    # simulation
    println("Simulation started")
    h_prior = h[end].data
    w_prior = w.data
    τ_prior = τ.data
    η_prior = η.data

    simulated_X = [X_test[1]]
    simulated_Y = [Y_test[1]]
    simulated_Y_cov = [0.0]
    simulated_error = Vector{Any}([h[end]])

    ProgressMeter.@showprogress for i in 1:length(Y_test)

    #     w_post, e_post, η_post, τ_post = inference_callback(h_prior, η_prior, τ_prior, w_prior, simulated_Y[i], simulated_X[i], U_test[i], delay_e, full_order, ϕ_, approximation)
        
    #     h_prior = MvNormalMeanCovariance([mean(e_post); mean(h_prior)[2:end]], Diagonal([cov(e_post); var(h_prior)[2:end]]))
    #     η_prior = η_post
    #     τ_prior = τ_post
    #     w_prior = w_post
        
        push!(simulated_X, [simulated_Y[i]; simulated_X[i][1:delay_y-1]])
        
        pred_sim = prediction(h_prior, mean(w_prior), η_prior, τ_prior, simulated_X[end], U_test[i], full_order=full_order, meta=NonlinearMeta(approximation, ϕ_, simulated_X[end], U_test[i]))

        push!(simulated_Y, mean(pred_sim))
        push!(simulated_Y_cov, var(pred_sim))
        push!(simulated_error, h_prior)
        

    end

    RMSE_sim = sqrt(mean((simulated_Y[2:end] .- Y_test).^2))
    
    priors_fl = Dict("θ" => (zeros(full_order,), Matrix{Float64}(I,full_order,full_order)), 
                     "τ" => (1.0, 1.0))

    PΦ = gen_combs(options)
    ϕ_fl(x::Array{Float64,1}) = [prod(x.^PΦ[:,k]) for k = 1:size(PΦ,2)]

    rms_sim_fl, rms_pred_fl, sim_fl, pred_fl, coefs_fl = ForneyNarmax.experiment_FEM(input_trn[1:n_train], output_trn[1:n_train], input_val[1:n_test], output_val[1:n_test], 
                                                                        ϕ_fl, priors_fl, M1=delay_u, M2=delay_y, M3=delay_e, N=full_order, num_iters=20, computeFE=false)

    # Specify which information should be saved in JLD2 file
    return @strdict experiment_params result_inf syn_noise η_true Y_test U_test X_test Y_train U_train X_train RMSE_pred RMSE_sim rms_sim_fl rms_pred_fl sim_fl pred_fl coefs_fl
end


results = map(experiments) do experiment
    # Path for the saving cache file for later use
    cache_path  = projectdir("dump", "narmax")
    # Types which should be used for cache file name
    save_types  = (String, Real, ET, UT)
    try
        result, _ = produce_or_load(cache_path, experiment, allowedtypes = save_types) do params
            run_experiment(params)
        end
        # generate_plots(result, "tikz")
        # generate_plots(result, "svg")

        return result
    catch error
        @warn error
    end
end