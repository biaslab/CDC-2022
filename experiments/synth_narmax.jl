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
    "delay"      => collect(1:4),
    "poly_order" => collect(2:3),
    "iterations" => 100,
    "w_true"     => [1e1, 1e2, 1e3, 1e4, 2e4],
    "seed"       => 42
))


function run_experiment(experiment_params)

    # We unpack all provided parameters into separate variables
    @unpack n_train, n_test, delay, poly_order, iterations, w_true, seed = experiment_params

    delay_u, delay_y, delay_e = fill(delay, 3)

    order_u, order_h = delay_u+1, delay_e+1
    
    options = Dict("na"=>delay_y, "nb"=>delay_u, "ne"=>delay_e, "nd"=>poly_order, "dc"=>true, "crossTerms"=>true, "noiseCrossTerms"=>false)

    function phi(options)
        
        precompiled = precompiled_phi(options)
        
        return (u, y, h) -> begin
            na = length(y)
            nb = length(u)-1
            ne = length(h)-1
            precompiled([u; y; h[2:end]])
        end 
    end


    
    full_order = length(ϕ(randn(sum([delay_u, delay_y, order_h])), options))
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

    h_prior, w_prior = MvNormalMeanPrecision(zeros(order_h), diageye(order_h)), GammaShapeRate(1e-4, 1e-4)
    η_prior, τ_prior = MvNormalMeanPrecision(zeros(full_order), diageye(full_order)),  GammaShapeRate(1e-2, 1e-2)
    
    narmax_imessages = (h = MvNormalMeanPrecision(zeros(order_h), 1e-10*diageye(order_h)), );

    narmax_imarginals = (h = h_prior,
                         w = w_prior,
                         θ = MvNormalMeanPrecision(zeros(order_h), 1e12*diageye(order_h)),
                         τ = τ_prior,
                         η = η_prior);

    narmax_constraints = @constraints begin
        q(θ) :: Marginal(MvNormalMeanPrecision(zeros(order_h), 1e12*diageye(order_h)))
        q(ẑ, z, η, τ, h_0, h, θ, w) = q(ẑ, z)q(η)q(τ)q(h_0, h)q(θ)q(w)
    end;

    ϕ_ = phi(options)
    narmax_model = Model(narmax, length(Y_train), 
                    (mean(h_prior), precision(h_prior)), 
                    (shape(w_prior), rate(w_prior)), 
                    (mean(η_prior), precision(η_prior)), 
                    (shape(τ_prior), rate(τ_prior)), 
                    X_train, U_train, order_h, full_order, ϕ_);
    



    println("Inference started")   
   
    # First execution is slow due to Julia's init compilation 
    result_inf = inference(
        model = narmax_model, 
        data  = (y = Y_train, ),
        constraints   = narmax_constraints,
        meta          = narmax_meta(Multivariate, order_h, full_order, ARsafe()),
        options       = model_options(limit_stack_depth = 500),
        initmarginals = narmax_imarginals,
        initmessages  = narmax_imessages,
        returnvars    = (θ = KeepLast(), w=KeepLast(), h=KeepLast(), η=KeepLast(), τ=KeepLast(), z=KeepLast(), ẑ=KeepLast()),
        free_energy   = true,
        iterations    = 100, 
        showprogress  = true
    );

    @unpack θ, w, h, η, τ, z, ẑ = result_inf.posteriors
    println("Prediction started")              
    # prediction
    predictions = []
    h_prior = h[end].data
    w_prior = w.data
    τ_prior = τ.data
    η_prior = η.data
    ProgressMeter.@showprogress for i in 1:length(Y_test)
        pred = prediction(h_prior, mean(w_prior), η_prior, τ_prior, full_order, order_h, meta=NonlinearMeta(ϕ_, X_test[i], U_test[i], 42))
        push!(predictions, pred)
        θ_post, w_post, h_post, η_post, τ_post = inference_callback(h_prior, η_prior, τ_prior, w_prior, [Y_test[i]], [X_test[i]], [U_test[i]], order_h, full_order, ϕ_)
        h_prior = h_post
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
        θ_post, w_post, h_post, η_post, τ_post = inference_callback(h_prior, η_prior, τ_prior, w_prior, [simulated_Y[i]], [simulated_X[i]], [U_test[i]], order_h, full_order, ϕ_)
        
        h_prior = h_post
        η_prior = η_post
        τ_prior = τ_post
        w_prior = w_post
        
        push!(simulated_X, [simulated_Y[i]; simulated_X[i][1:delay_y-1]])
        
        pred_sim = prediction(h_prior, mean(w_prior), η_prior, τ_prior, full_order, order_h, meta=NonlinearMeta(ϕ_, simulated_X[end], U_test[i], 42))

        push!(simulated_Y, mean(pred_sim))
        push!(simulated_Y_cov, var(pred_sim))
        push!(simulated_error, h_prior)
    end

    RMSE_sim = sqrt(mean((simulated_Y[2:end] .- Y_test).^2))
    # Specify which information should be saved in JLD2 file
    return @strdict experiment_params result_inf syn_noise η_true Y_test U_test X_test Y_train U_train X_train RMSE_pred RMSE_sim
end


results = map(experiments) do experiment
    # Path for the saving cache file for later use
    cache_path  = projectdir("dump", "narmax")
    # Types which should be used for cache file name
    save_types  = (String, Real)
    result, _ = produce_or_load(cache_path, experiment, allowedtypes = save_types) do params
        run_experiment(params)
    end
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