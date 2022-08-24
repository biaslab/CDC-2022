using DrWatson
@quickactivate :NARMAXExperiments

using Distributions
using GraphPPL
using ReactiveMP
using Random
using Plots
using LinearAlgebra
using StatsBase
using DataStructures
import ProgressMeter
using MAT

pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}");

# We create a bunch of experiments with `dict_list` function from DrWatson.jl package
# This creates a list of parameters for each experiment, we use `@onlyif` to restrict some parameters being dependant on others

train_sizes =  [100, 200, 400, 800]
n_sim = 100

experiments = dict_list(Dict(
    "type"       => "narmax",
    "n_train"    => train_sizes,
    "n_test"     => 1000,
    "delay"      => 1,
    "poly_order" => 3,
    "iterations" => 50,
    "approximation" => UT(),
    "seed"       => collect(1:n_sim)
))

function run_experiment(experiment_params)

    # We unpack all provided parameters into separate variables
    @unpack n_train, n_test, delay, poly_order, iterations, seed, approximation = experiment_params

    delay_u, delay_y, delay_e = fill(delay, 3)
    order_u = delay_u + 1
    
    options = Dict("na"=>delay_y, "nb"=>delay_u, "ne"=>delay_e, "nd"=>poly_order, "dc"=>false, "crossTerms"=>true, "noiseCrossTerms"=>false)

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

    # Load datasets
    mat_data = matread("datasets/verification/realizations/NARMAXsignal_stde0.05_degree3_delays4_D22_r$seed.mat")
    train_size = n_train
    test_size = n_test

    # Length of transient period
    transient = 0

    # Length of training signal
    ix_trn = collect(1:train_size) .+ transient

    # Length of testing signal
    ix_tst = collect(1:test_size) .+ transient;

    # Extract data sets
    input_trn = mat_data["uTrain"][ix_trn]
    noise_trn = mat_data["eTrain"][ix_trn]
    output_trn = mat_data["yTrain"][ix_trn]

    input_tst = mat_data["uTest"][ix_tst]
    noise_tst = mat_data["eTest"][ix_tst]
    output_tst = mat_data["yTest"][ix_tst]

    # System parameters
    η_true = mat_data["system"]["theta"][:]
    γ_true = inv(mat_data["options"]["stde"]^2);

    # Generate training data
    observations_prev, observations = ssm(output_trn, delay_y)
    controls = ssm(input_trn, order_u)[1]
    X_train, Y_train, U_train = observations_prev[1:train_size-order_u], observations[1:train_size-order_u], controls[1:train_size-order_u];

    # Generate validation data
    observations_prev, observations = ssm(output_tst, delay_y)
    controls = ssm(input_tst, order_u)[1]
    X_test, Y_test, U_test = observations_prev[1:test_size-order_u], observations[1:test_size-order_u], controls[1:test_size-order_u];

    w_prior, τ_prior =  GammaShapeRate(1.0, 1.0), GammaShapeRate(1e2, 1.0)
    h_prior, η_prior = MvNormalMeanPrecision(zeros(delay_e), diageye(delay_e)),  MvNormalMeanPrecision(zeros(full_order), diageye(full_order))
    
    narmax_imessages = (e = NormalMeanPrecision(0.0, var(Y_train)), );

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
        returnvars    = (w=KeepLast(), h=KeepLast(), e=KeepLast(), η=KeepLast(), τ=KeepLast(), z=KeepLast(), ẑ=KeepLast()),
        free_energy   = true,
        iterations    = 50, 
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
        pred = prediction(h_prior, mean(w_prior), η_prior, τ_prior, full_order=full_order, meta=NonlinearMeta(approximation, ϕ_, X_test[i], U_test[i]))
        push!(predictions, pred)
        w_post, e_post, η_post, τ_post, h_post = inference_callback(h_prior, η_prior, τ_prior, w_prior, Y_test[i], X_test[i], U_test[i], delay_e, full_order, ϕ_, approximation)
        h_prior = h_post
        η_prior = η_post
        τ_prior = τ_post
        w_prior = w_post
    end

    RMSE_pred = sqrt(mean((mean.(predictions) .- Y_test) .^2))

    
    # simulation
    println("Simulation started")

    h_prior = MvNormalMeanPrecision(zeros(delay_e), diageye(delay_e))
    w_prior = w.data
    τ_prior = τ.data
    η_prior = η.data

    simulated_X = [X_test[1]]
    simulated_Y = [NormalMeanPrecision(Y_test[1], 1e4) for _ in 1:delay_y]
    simulated_error = Vector{Any}([h_prior])

    ProgressMeter.@showprogress for i in 1:length(Y_test)
        
        push!(simulated_X, [mean(simulated_Y[i]); simulated_X[i][1:delay_y-1]])
        msg_y = MvNormalMeanPrecision(mean.(simulated_Y[end-delay_y+1:end]), Diagonal(var.(simulated_Y[end-delay_y+1:end])))
        pred_sim = prediction_(h_prior, msg_y, mean(w_prior), η_prior, τ_prior, full_order=full_order, meta=NonlinearMeta(UT(), ϕ_, X_test[i], U_test[i]))
        push!(simulated_Y, pred_sim)
        push!(simulated_error, h_prior)
        

    end
    RMSE_sim = sqrt(mean((mean.(simulated_Y[delay_y+1:end]) .- Y_test).^2))
    
    priors_fl = Dict("θ" => (zeros(full_order,), Matrix{Float64}(I,full_order,full_order)), 
                     "τ" => (1.0, 1.0))

    PΦ = gen_combs(options)
    ϕ_fl(x::Array{Float64,1}) = [prod(x.^PΦ[:,k]) for k = 1:size(PΦ,2)]

    rms_sim_fl, rms_pred_fl, sim_fl, pred_fl, coefs_fl, residuals_fl = ForneyNarmax.experiment_FEM(input_trn[1:n_train], output_trn[1:n_train], input_tst[1:n_test], output_tst[1:n_test], ϕ_fl, priors_fl, M1=delay_u, M2=delay_y, M3=delay_e, N=full_order, num_iters=20, computeFE=false)

    # Specify which information should be saved in JLD2 file
    return @strdict experiment_params result_inf η_true γ_true input_trn output_trn noise_trn input_tst output_tst noise_tst RMSE_pred RMSE_sim coefs_fl rms_sim_fl rms_pred_fl sim_fl pred_fl residuals_fl predictions simulated_Y
end


results = map(experiments) do experiment
    # Path for the saving cache file for later use
    cache_path  = projectdir("dump", "narmax")
    # Types which should be used for cache file name
    save_types  = (String, Real, ET, UT)
    try
        result, _ = produce_or_load(cache_path, experiment, allowedtypes = save_types, force=true) do params
            run_experiment(params)
        end
        # generate_plots(result, "tikz")
        # generate_plots(result, "svg")

        return result
    catch error
        @error error
    end

    return result
end


### FE

begin
    FE = []
    for res in results
        try
            push!(FE, res["result_inf"].free_energy)
        catch error
            @warn error
        end
    end
    FE = sum(FE)/length(FE)
    fig_path = "experiments/results/synthetic/FE.tikz"
    vmp_its = experiments[1]["iterations"]
    pfe = plot(1:vmp_its, FE, ylabel="Bethe free energy [nats]", color=:red, markershape=:circle, markersize=4, xlabel="iteration", legend=:topright, label="BFE", ylabelfontsize=13, xlabelfontsize=13, legendfontsize=10, markeralpha=0.3, xtickfontsize=10, ytickfontsize=10, size=(450, 300))
    savefig(pfe, fig_path)
end



begin
    ### RMSE
    n_trains = train_sizes
    rmses  = repeat([[]], length(n_trains))

    RMSE_sim_UT = SortedDict(n_trains .=> deepcopy.(rmses))
    RMSE_sim_ACC = SortedDict(n_trains .=> deepcopy.(rmses))
    RMSE_sim_ILS = SortedDict(n_trains .=> deepcopy.(rmses))

    RMSE_pred_UT = SortedDict(n_trains .=> deepcopy.(rmses))
    RMSE_pred_ACC = SortedDict(n_trains .=> deepcopy.(rmses))
    RMSE_pred_ILS = SortedDict(n_trains .=> deepcopy.(rmses))

    # fill up results from julia code
    for i in 1:length(results)
        try
            if typeof(results[i]["experiment_params"]["approximation"]) == UT
                n_train = results[i]["experiment_params"]["n_train"]
                push!(RMSE_sim_UT[n_train], results[i]["RMSE_sim"])
                push!(RMSE_pred_UT[n_train], results[i]["RMSE_pred"])

                push!(RMSE_sim_ACC[n_train], results[i]["rms_sim_fl"])
                push!(RMSE_pred_ACC[n_train], results[i]["rms_pred_fl"])
            end
        catch error
            @warn error
        end
        
    end

    # fill up results from MATLAB
    for seed in 1:n_sim
        sim = SortedDict(train_sizes .=> matread("experiments/results/results-NARMAX_ILS_stde0.05_pol3_delays4_D22_degree3_r$seed.mat")["RMS_sim"]')
        pred = SortedDict(train_sizes .=> matread("experiments/results/results-NARMAX_ILS_stde0.05_pol3_delays4_D22_degree3_r$seed.mat")["RMS_prd"]')
        for key in keys(sim)
            @show key
            push!(RMSE_pred_ILS[key], pred[key])
            push!(RMSE_sim_ILS[key], sim[key])
        end
    end

end

function mean_std_rmse(sdict)
    map(collect(sdict)) do pair 
        key = pair[1]
        vals = pair[2]
        
        vals = filter(e -> !isnan(e) && !isinf(e) && isless(e, 1.0), vals)
        return key => (mean(vals), std(vals)/sqrt(length(vals)))
    end |> SortedDict
end

function failed(sdict)
    map(collect(sdict)) do pair 
        key = pair[1]
        vals = pair[2]
        
        vals = map(e -> isnan(e) || isinf(e) || !isless(e, 1.0), vals)
        return key => sum(vals)
    end |> SortedDict
end

# training <-> failures
begin
    failed_sim_acc = failed(RMSE_sim_ACC)
    failed_sim_ut  = failed(RMSE_sim_UT)
    failed_sim_ils = failed(RMSE_sim_ILS)

    ps = plot(train_sizes, first.(collect(values(failed_sim_ut)))./n_sim, label="fVB", xlabel="length of training signal", markershape=:circle, fillalpha=0.2, markersize=4, color=:black, width=3, alpha=0.7)
    plot!(train_sizes, first.(collect(values(failed_sim_acc)))./n_sim, label="cVB", title="simulation", xticks=train_sizes, markershape=:circle, ylabel="proportion failed", fillalpha=0.2, markersize=4, color=:red, width=3, alpha=0.7)
    plot!(train_sizes, first.(collect(values(failed_sim_ils)))./n_sim, label="ILS", xlabel="length of training signal", markershape=:circle, fillalpha=0.2, markersize=4, legend=:topright, size=(600, 300), ylabelfontsize=13, xlabelfontsize=13, legendfontsize=10, markeralpha=0.3, xtickfontsize=10, ytickfontsize=10, color=:purple, width=3, alpha=0.7)
    savefig(ps, "experiments/results/synthetic/simulation_failed.tikz")

    failed_pred_acc = failed(RMSE_pred_ACC)
    failed_pred_ut  = failed(RMSE_pred_UT)
    failed_pred_ils = failed(RMSE_pred_ILS)
    pp = plot(train_sizes, first.(collect(values(failed_pred_ut)))./n_sim, label="fVB", markershape=:circle, ylabel="proportion failed", fillalpha=0.2, markersize=4, color=:black, width=3, alpha=0.7)
    plot!(train_sizes, first.(collect(values(failed_pred_acc)))./n_sim, label="cVB", title="prediction", xticks=train_sizes, markershape=:circle, fillalpha=0.2, markersize=4, color=:red, width=3, alpha=0.7)
    plot!(train_sizes, first.(collect(values(failed_pred_ils)))./n_sim, label="ILS", xlabel="length of training signal", markershape=:circle, fillalpha=0.2, markersize=4, legend=:topright, size=(600, 300), ylabelfontsize=13, xlabelfontsize=13, legendfontsize=10, markeralpha=0.8, xtickfontsize=10, ytickfontsize=10, color=:purple, width=3, alpha=0.7)
    savefig(pp, "experiments/results/synthetic/prediction_failed.tikz")
end


# training <-> RMSE
begin
    rmse_sim_acc = mean_std_rmse(RMSE_sim_ACC)
    rmse_sim_ut = mean_std_rmse(RMSE_sim_UT)
    rmse_sim_ils = mean_std_rmse(RMSE_sim_ILS)

    ps = plot(train_sizes, first.(collect(values(rmse_sim_ut))), ribbon=(last.(collect(values(rmse_sim_ut)))), label="fVB", xlabel="length of training signal", markershape=:circle, fillalpha=0.3, markersize=4, color=:black, width=2, alpha=0.7)
    plot!(train_sizes, first.(collect(values(rmse_sim_acc))), ribbon=(last.(collect(values(rmse_sim_acc)))), label="cVB", title="simulation", xticks=train_sizes, markershape=:circle, ylabel="RMS", fillalpha=0.3, markersize=3, color=:red, width=2, alpha=0.7)
    plot!(train_sizes, first.(collect(values(rmse_sim_ils))), ribbon=(last.(collect(values(rmse_sim_ils)))), label="ILS", xlabel="length of training signal", markershape=:circle, fillalpha=0.3, markersize=4, color=:purple, width=2, alpha=0.7, legend=:topright, size=(600, 300),ylabelfontsize=13, xlabelfontsize=13, legendfontsize=10, markeralpha=0.4, xtickfontsize=10, ytickfontsize=10)
    savefig(ps, "experiments/results/synthetic/simulation_rms.tikz")

    rmse_pred_acc = mean_std_rmse(RMSE_pred_ACC)
    rmse_pred_ut = mean_std_rmse(RMSE_pred_UT)
    rmse_pred_ils = mean_std_rmse(RMSE_pred_ILS)
    pp = plot(train_sizes, first.(collect(values(rmse_pred_ut))), ribbon=(last.(collect(values(rmse_pred_ut)))), label="fVB", markershape=:circle, ylabel="RMS", fillalpha=0.3, markersize=4, color=:black, width=2, alpha=0.7)
    plot!(train_sizes, first.(collect(values(rmse_pred_acc))), ribbon=(last.(collect(values(rmse_pred_acc)))), label="cVB", title="prediction", xticks=train_sizes, markershape=:circle, ylabel="RMS", fillalpha=0.2, markersize=4, color=:red, width=3, alpha=0.7)
    plot!(train_sizes, first.(collect(values(rmse_pred_ils))), ribbon=(last.(collect(values(rmse_pred_ils)))), label="ILS", ylims=(0.05, 0.1), xlabel="length of training signal", markershape=:circle, fillalpha=0.2, legend=:topright, size=(600, 300),ylabelfontsize=13, xlabelfontsize=13, legendfontsize=10, markeralpha=0.3, xtickfontsize=10, ytickfontsize=10, markersize=4, color=:purple, width=3, alpha=0.7)
    savefig(pp, "experiments/results/synthetic/prediction_rms.tikz") 
end

# inference results
# errors tracking
#FIXME: This function should be removed, the errors must be obtained from results
function get_train_errors(pick, n_train=100, n_test=100)
    options = Dict("na"=>1, "nb"=>1, "ne"=>1, "nd"=>3, "dc"=>false, "crossTerms"=>true, "noiseCrossTerms"=>false)

    full_order = length(ϕ(zeros(1+1+1+1), options))

    # Load datasets
    mat_data = matread("datasets/verification/realizations/NARMAXsignal_stde0.05_degree3_delays4_D22_r$pick.mat")
    train_size = n_train
    test_size = n_test

    # Length of transient period
    transient = 0

    # Length of training signal
    ix_trn = collect(1:train_size) .+ transient

    # Length of testing signal
    ix_tst = collect(1:test_size) .+ transient;

    # Extract data sets
    input_trn = mat_data["uTrain"][ix_trn]
    output_trn = mat_data["yTrain"][ix_trn]

    input_tst = mat_data["uTest"][ix_tst]
    output_tst = mat_data["yTest"][ix_tst]

    priors_fl = Dict("θ" => (zeros(full_order,), Matrix{Float64}(I,full_order,full_order)), 
                     "τ" => (1.0, 1.0))

    PΦ = gen_combs(options)
    ϕ_fl(x::Array{Float64,1}) = [prod(x.^PΦ[:,k]) for k = 1:size(PΦ,2)]

    _, _, _, _, _, residuals_fl = ForneyNarmax.experiment_FEM(input_trn[1:n_train], output_trn[1:n_train], input_tst[1:n_test], output_tst[1:n_test], ϕ_fl, priors_fl, M1=1, M2=1, M3=1, N=full_order, num_iters=20, computeFE=false)

    return residuals_fl
end
begin
    xlims = (0, 100)
    pick = 11
    example_res = results[pick]
    inf_errors = mean.(example_res["result_inf"].posteriors[:e]), var.(example_res["result_inf"].posteriors[:e])
    err_len = length(inf_errors[1])
    residuals = get_train_errors(pick) #example_res["residuals_fl"]
    noises = example_res["noise_trn"]
    pe = plot(inf_errors[1], ribbon=sqrt.(inf_errors[2]),linestyle=:dash, width=3, label="inferred")
    plot!(noises[2:err_len+1], width=1.5, label="generated", xlims=xlims, size=(600, 300), ylabelfontsize=13, xlabelfontsize=13, legendfontsize=10, markeralpha=0.4, xtickfontsize=10, ytickfontsize=10, legend=:topright, xlabel="k", ylabel="amplitude", title="")
    savefig(pe, "experiments/results/synthetic/fVB_errors.tikz")

    pe_ = plot(residuals[2:end], linestyle=:dot, width=3, label="estimated")
    plot!(noises[1:end], width=1.5, label="generated", xlims=xlims, size=(600, 300), ylabelfontsize=13, xlabelfontsize=13, legendfontsize=10, markeralpha=0.4, xtickfontsize=10, ytickfontsize=10, legend=:topright, xlabel="k", ylabel="amplitude", title="",)
    savefig(pe_, "experiments/results/synthetic/cVB_errors.tikz")
end