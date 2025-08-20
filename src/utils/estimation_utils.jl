# using LatinHypercubeSampling, Accessors, Optim, LineSearches, Dates
using LatinHypercubeSampling, Accessors, Optim, LineSearches

function unpack(sim_output::SimulationOutput)
	# unpack the simulation output
	sim_params = sim_output.sim_params
	sim_hyper = sim_params.sim_hyper

	return sim_params, sim_hyper
end
function unpack(est_output::EstimationOutput)
	sim_output = est_output.sim_output
	sim_params = sim_output.sim_params
	sim_hyper = sim_params.sim_hyper

	return sim_output, sim_params, sim_hyper
end

function model_params_2_param_vec(
	model_params::NamedTuple;
	ini_ref_state = 1, tr_ref_state = 1
)
	# convert model params to a parameter vector

	# remove reference vector
	beta_initial = model_params.beta_initial[1:end .!= ini_ref_state, :]
	beta_transition = model_params.beta_transition[1:end .!= tr_ref_state, :, :]

	# switch estimates to param vector
	beta_initial_vec = vec(beta_initial)
	beta_transition_vec = vec(beta_transition)
	rho_initial_vec = haskey(model_params, :rho_initial) ? vec(model_params.rho_initial) : Float64[]
	rho_trans_vec = haskey(model_params, :rho_trans) ? vec(model_params.rho_trans) : Float64[]
	beta_bernoulli_vec = haskey(model_params.emissions, :beta_bernoulli) ? vec(model_params.emissions.beta_bernoulli) : Float64[]
	gaussian_means_vec = haskey(model_params.emissions, :beta_gaussian) ? vec(model_params.emissions.beta_gaussian.means) : Float64[]
	gaussian_stds_vec = haskey(model_params.emissions, :beta_gaussian) ? vec(model_params.emissions.beta_gaussian.stds) : Float64[]

	out = vcat(
		beta_initial_vec, 
		beta_transition_vec, 
		rho_initial_vec, 
		rho_trans_vec, 
		beta_bernoulli_vec, 
		gaussian_means_vec, 
		gaussian_stds_vec
		)

	return out
end

function param_vec_2_model_params(
	params_vec::Vector{<:Real};
	n_states::Int64,
	n_obs_tup::NamedTuple,
	covariate_tup::NamedTuple,
	ini_ref_state = 1, tr_ref_state = 1
)

	set_k_from_rho_initial = length(covariate_tup[:initial]) == 0 ? 1 : 2
	set_k_from_rho_trans = length(covariate_tup[:trans]) == 0 ? 1 : 2

	dims_initial = (n_states - 1, set_k_from_rho_initial); l_initial = prod(dims_initial)
	dims_trans = (n_states - 1, set_k_from_rho_trans, n_states); l_trans = prod(dims_trans)
	dims_rho_initial = (length(covariate_tup[:initial])); l_rho_initial = prod(dims_rho_initial); 
	dims_rho_trans = (length(covariate_tup[:trans])); l_rho_trans = prod(dims_rho_trans); 

	dims_beta_bern = haskey(n_obs_tup, :bernoulli) ? (n_states, length(covariate_tup[:em]) + 1, n_obs_tup.bernoulli) : (); l_bern = prod(dims_beta_bern)
	dims_beta_gauss_means = haskey(n_obs_tup, :gaussian) ? (n_states, n_obs_tup.gaussian) : (); l_gauss = prod(dims_beta_gauss_means)

	idx = 1
	beta_initial = reshape(params_vec[idx:idx+l_initial-1], dims_initial); idx += l_initial
	beta_transition = reshape(params_vec[idx:idx+l_trans-1], dims_trans); idx += l_trans

	# add the reference vectors
	beta_initial = cat(beta_initial[1:(ini_ref_state-1), :], zeros(1, set_k_from_rho_initial), beta_initial[ini_ref_state:end, :], dims = 1)
	beta_transition = cat(beta_transition[1:(tr_ref_state-1), :, :], zeros(1, set_k_from_rho_trans, n_states), beta_transition[tr_ref_state:end, :, :], dims = 1)

	rho_initial = reshape(params_vec[idx:idx+l_rho_initial-1], dims_rho_initial); idx += l_rho_initial
	rho_trans = reshape(params_vec[idx:idx+l_rho_trans-1], dims_rho_trans); idx += l_rho_trans

	emissions = NamedTuple()
	if haskey(n_obs_tup, :bernoulli)
		dims_beta_bern = (n_states, length(covariate_tup[:em]) + 1, n_obs_tup.bernoulli)
		l_bern = prod(dims_beta_bern)
		beta_bernoulli = reshape(params_vec[idx:idx+l_bern-1], dims_beta_bern); idx += l_bern
		emissions = (; emissions..., beta_bernoulli = beta_bernoulli)
	end

	if haskey(n_obs_tup, :gaussian)
		dims_beta_gauss_means = (n_states, n_obs_tup.gaussian)
		l_gauss = prod(dims_beta_gauss_means)
		beta_gaussian_means = reshape(params_vec[idx:idx+l_gauss-1], dims_beta_gauss_means); idx += l_gauss
		beta_gaussian_stds = reshape(params_vec[idx:idx+l_gauss-1], dims_beta_gauss_means); idx += l_gauss
		beta_gaussian = (
			means = beta_gaussian_means,
			stds = beta_gaussian_stds
		)	
		emissions = (; emissions..., beta_gaussian = beta_gaussian)
	end

	# beta_bernoulli = reshape(params_vec[idx:idx+l_bern-1], dims_beta_bern); idx += l_bern
	# beta_gaussian_means = reshape(params_vec[idx:idx+l_gauss-1], dims_beta_gauss_means); idx += l_gauss
	# beta_gaussian_stds = reshape(params_vec[idx:idx+l_gauss-1], dims_beta_gauss_means); idx += l_gauss

	# emissions = NamedTuple()
	# for dist in keys(n_obs_tup)
	# 	if dist == :bernoulli
	# 		emissions = (; emissions..., beta_bernoulli = beta_bernoulli)
	# 	elseif dist == :gaussian
	# 		beta_gaussian = (
	# 			means = beta_gaussian_means,
	# 			stds = beta_gaussian_stds
	# 		)	
	# 		emissions = (; emissions..., beta_gaussian = beta_gaussian)
	# 	end
	# end

	model_params = (
		beta_initial = beta_initial,
		beta_transition = beta_transition,
		rho_initial = rho_initial,
		rho_trans = rho_trans,
		emissions = emissions
	)

	return model_params
end

function objective_func(
	params_vec::Vector{<:Real};
	sim_output::SimulationOutput,
	lambda::Float64 = 0.0
)
	sim_params, sim_hyper = unpack(sim_output)

	model_params = param_vec_2_model_params(
		params_vec;
		n_states = sim_hyper.n_states,
		n_obs_tup = sim_hyper.n_obs_tup,
		covariate_tup = sim_hyper.covariate_tup
	)

	lkl = lkl_func(;
		model_params = model_params,
		sim_output = sim_output,
	)
	penalised_lkl = lkl

	# penalisation
	if lambda > 0

		if haskey(model_params, :rho_initial)
			penalised_lkl = penalised_lkl + lambda * (norm(model_params.rho_initial) - 1)^2
		end

		if haskey(model_params, :rho_trans)
			penalised_lkl = penalised_lkl + lambda * (norm(model_params.rho_trans) - 1)^2
		end

		if haskey(model_params.emissions, :beta_gaussian)
			true_gaussian_emissions = convert_gauss_2_true(model_params.emissions.beta_gaussian)
			variances = true_gaussian_emissions.stds[:, :].^2
			em_var_pen =  (1 / sim_hyper.N) * sum(variances.^(-2)) + sum(log.(variances))
			penalised_lkl = penalised_lkl + em_var_pen

			# penalise the mean too for better convergences
			means = true_gaussian_emissions.means[:, :].^2
			em_means_pen =  (1 / sim_hyper.N) * sum(means.^(-2)) + sum(log.(means))
			penalised_lkl = penalised_lkl + em_means_pen
		end
	end

	return penalised_lkl
end

# run est module
function run_estimation(
	sim_output::SimulationOutput;
	est_seed = 1,
	lambda = 0.0,
	meth = "GradientDescent",
	n_starts = 10,
	starting_model_params = nothing,
	calculate_hessian = false,
	kwargs...
)

	# unpack the structures
	sim_params, sim_hyper = unpack(sim_output)

	# handle missing covariates -- skip for now
	# expanded_patients, expanded_observations, expanded_observations_cont, lkl_weights = expand_missing_covariate_entries(sim_output)

	# define the likelihood function
	obj_func = params_vec -> objective_func(
		params_vec,

		sim_output = sim_output,
		lambda = lambda
	)

	# true parameter info
	true_parameter_vec = model_params_2_param_vec(
		sim_params.model_params;
		ini_ref_state = sim_hyper.ref_state, tr_ref_state = sim_hyper.ref_state
	)
	true_log_lkl = obj_func(true_parameter_vec)

	# generate starting params
	starting_params_vec = generate_starting_params_vec(
		sim_output;
		n_starts = n_starts,
		starting_model_params = starting_model_params,
		est_seed = est_seed
	)

	# quick estimation
	print("\nEstimating\n")
	opt_list = run_optimisation(
		obj_func;
		meth = meth,
		starting_params_vec = starting_params_vec,
		kwargs... # add kwargs if needed
	)

	# get the starting model params
	starting_model_params = param_vec_2_model_params.(
		starting_params_vec;
		n_states = sim_hyper.n_states,
		n_obs_tup = sim_hyper.n_obs_tup,
		covariate_tup = sim_hyper.covariate_tup
	)

	# fitted model params
	fitted_model_params = param_vec_2_model_params.(
		Optim.minimizer.(opt_list);
		n_states = sim_hyper.n_states,
		n_obs_tup = sim_hyper.n_obs_tup,
		covariate_tup = sim_hyper.covariate_tup
	)

	# # get hessian
	if calculate_hessian
		print("\nGetting Hessian\n")
		flush(stdout)
		temp_func = (params -> get_hessian(obj_func, params))
		hessian_list = temp_func.(Optim.minimizer.(opt_list))
	else
		print("\nWARNING: Hessian not calculated\n")
		flush(stdout)
		hessian_list = [zeros(length(starting_params_vec[1]), length(starting_params_vec[1])) for i in 1:length(starting_params_vec)]
	end
	se_vec = get_se.(hessian_list)
	se_fitted_model_params = param_vec_2_model_params.(
		se_vec;
		n_states = sim_hyper.n_states,
		n_obs_tup = sim_hyper.n_obs_tup,
		covariate_tup = sim_hyper.covariate_tup
	)

	# # construct results
	est_output = EstimationOutput(
		sim_output = sim_output,
		est_seed = est_seed,
		lambda = lambda, 

		# true_parameter_vec = true_parameter_vec,
		true_log_lkl = true_log_lkl, 

		starting_model_params = starting_model_params,
		fitted_model_params = fitted_model_params,

		# starting_params_vec = starting_params_vec,
		# fitted_params_vec = Optim.minimizer.(opt_list),
		fitted_log_lkl_list = Optim.minimum.(opt_list),
		se_fitted_model_params = se_fitted_model_params,

		iterations = Optim.iterations.(opt_list),
		iteration_limit_reached = Optim.iteration_limit_reached.(opt_list),
		converged = Optim.converged.(opt_list)
		
	)

	return est_output
end

function run_optimisation(
	obj_func;
	meth = "BFGS",
	starting_params_vec::Vector{Vector{Float64}},
	kwargs...
)

	n_starts = length(starting_params_vec)
	# using multi threading
	if n_starts > Threads.nthreads()
		# appropriate split threads
		start_chunks_idx = Iterators.partition(1:n_starts, n_starts ÷ Threads.nthreads())
	else
		start_chunks_idx = Iterators.partition(1:n_starts, 1)
	end

	opt_list_ref = Array{Any}(undef, n_starts)
	opt_list_tasks = map(start_chunks_idx) do chunk_idx
		Threads.@spawn run_opt(
			obj_func;
			meth = meth,
			starting_params_vec = starting_params_vec[chunk_idx],
			kwargs... 
		)

	end
	opt_list = fetch.(opt_list_tasks)

	opt_idx = 1
	for i in 1:length(opt_list)
		for j in 1:length(opt_list[i])
			opt_list_ref[opt_idx] = opt_list[i][j]
			opt_idx +=1
		end
		
	end

	return opt_list_ref
end

function run_opt(
	obj_func;
	starting_params_vec::Vector{Vector{Float64}},
	meth = "BFGS",
	kwargs...
)

	default_opts = (
		show_every = 1,
		# store_trace = true,
		show_trace = true,
		iterations = 100,
		f_reltol = 1.0e-6, 
	)
	opts = merge(default_opts, kwargs)
	# opts = default_opts
	options_obj = Optim.Options(; opts...)

	opt_list = Array{Any}(undef, length(starting_params_vec))
	for i_start in 1:length(starting_params_vec)
		if meth == "GradientDescent"
			opt_list[i_start] = optimise_with_grad_desc(
				obj_func,
				starting_params_vec[i_start],
				options_obj;
				meth = GradientDescent(;
					linesearch = LineSearches.HagerZhang()),
			)
		elseif meth == "BFGS"
			opt_list[i_start] = optimise_with_bfgs(
				obj_func,
				starting_params_vec[i_start],
				options_obj;
				meth = BFGS(; linesearch = LineSearches.HagerZhang()),
			)
		else
			error("No opt method!")
		end

		# print("Done 1 Optimsation:\t"*string(Dates.canonicalize(now() - time_start))*"\n")
		flush(stdout)
	end

	return opt_list
end

function optimise_with_grad_desc(
	obj_func,
	starting_params,
	options_obj;
	meth = GradientDescent(;
		linesearch = LineSearches.HagerZhang()),
)
	# run optimisation with grad descent config
	opt = optimize(
		obj_func,
		starting_params,
		meth,
		options_obj,
		autodiff = :forward,
	)

	return opt
end

function optimise_with_bfgs(
	obj_func,
	starting_params,
	options_obj;
	meth = BFGS(;
		linesearch = LineSearches.HagerZhang()),
)
	# run optimisation with grad descent config

	opt = optimize(
		obj_func,
		starting_params,
		meth,
		options_obj,
		autodiff = :forward,
	)

	return opt
end

function generate_starting_params_vec(
	sim_output::SimulationOutput;
	n_starts::Int64 = 1,
	starting_model_params = nothing,
	est_seed = 1
)

	param_length = length(model_params_2_param_vec(sim_output.sim_params.model_params))
	if !isnothing(starting_model_params)
		starting_params_vec = [model_params_2_param_vec(starting_model_params)]
	else
		# generate starting params randomly
		if  n_starts == 1
			starting_params_vec = [rand(MersenneTwister(est_seed), Uniform(-2, 2), param_length)]
		elseif n_starts > 1
			# use latin hypercube for the multistart, with given estimation hyperparams
			plan, _ = LHCoptim(param_length, n_starts, 100; rng = MersenneTwister(est_seed))
			bounds_tuple = [(-2, 2) for _ in 1:param_length]
			mat = scaleLHC(plan, bounds_tuple)
			starting_params_vec = [vec(mat[:, i]) for i in 1:n_starts]
		end

		# replace params related to gauss with kmeans starting points
		adjusted_params_vec = [replace_gauss_params_with_kmeans(
			sim_output,
			starting_params_vec[i];
			seed = est_seed + i
		) for i in 1:length(starting_params_vec)]
		starting_params_vec = adjusted_params_vec
	end

	return starting_params_vec
end

function replace_gauss_params_with_kmeans(
	sim_output::SimulationOutput,
	param_vec::Vector{Float64};
	seed = 1,
	timepoint = 2 # timepoint to use for kmeans
)

	sim_params, sim_hyper = unpack(sim_output)
	if !haskey(sim_hyper.n_obs_tup, :gaussian)
		# if no gaussian params, return the original vector
		return param_vec
	else
		# if gaussian params exist, replace them with kmeans starting points
		new_gaussian_means = similar(sim_params.model_params.emissions.beta_gaussian.means)
		new_gaussian_stds = similar(sim_params.model_params.emissions.beta_gaussian.stds)
		for i in 1:sim_hyper.n_obs_tup.gaussian
			km_est = get_kmeans_estimates(sim_output.observations.gaussian_observations[:, timepoint, i], sim_hyper.n_states; seed = seed)

			new_gaussian_means[:, i] = km_est.means
			new_gaussian_stds[:, i] = sqrt.(km_est.variances)
		end

		# avoid numeric errors
		new_gaussian_means[new_gaussian_means .< 0] .= 0.01
		new_gaussian_means[new_gaussian_means .> 1] .= 0.99
		new_gaussian_stds[new_gaussian_stds .< 0] .= 0.01
		new_gaussian_stds[new_gaussian_stds .> 1] .= 0.99

		# convert to beta form and set the new params
		new_beta_gaussian = convert_gauss_2_form(
			(means = new_gaussian_means,
			stds = new_gaussian_stds)
		)

		new_model_params_temp = param_vec_2_model_params(
			param_vec;
			n_states = sim_hyper.n_states,
			n_obs_tup = sim_hyper.n_obs_tup,
			covariate_tup = sim_hyper.covariate_tup,
			ini_ref_state = sim_hyper.ref_state, tr_ref_state = sim_hyper.ref_state
		)

		new_model_params = deepcopy(new_model_params_temp)
		new_model_params = @set new_model_params.emissions.beta_gaussian = new_beta_gaussian
		
		new_param_vec = model_params_2_param_vec(
			new_model_params;
			ini_ref_state = sim_hyper.ref_state, tr_ref_state = sim_hyper.ref_state
		)

		return new_param_vec
	end
end

function get_kmeans_estimates(data::Vector{Float64}, k::Int; seed = 1)

	non_missing_data = data[data .!= -1]
    initial_means, assignments = simple_kmeans(non_missing_data, k; seed = seed)
    
    # Initialize variances and weights
    initial_variances = zeros(k)
    initial_weights = zeros(k)
    for i in 1:k
        cluster_data = non_missing_data[assignments .== i]
        if !isempty(cluster_data)
            initial_variances[i] = var(cluster_data, corrected=true)
            initial_weights[i] = length(cluster_data) / length(non_missing_data)
        else
            # Handle empty clusters
            initial_variances[i] = var(non_missing_data, corrected=true)
            initial_weights[i] = 1e-6
        end
    end
    
    # Normalize weights
    initial_weights /= sum(initial_weights)
    
    return (means = initial_means, variances = initial_variances, weights = initial_weights)
end

function simple_kmeans(data::Vector{Float64}, k::Int; max_iter::Int=100, seed=1)
    # Initialize cluster centers randomly
    means = sample(MersenneTwister(seed), data, k)
    assignments = zeros(Int, length(data))
    for iter in 1:max_iter
        # Assign data points to the nearest cluster
        for i in 1:length(data)
            distances = abs.(data[i] .- means)
            assignments[i] = argmin(distances)
        end
        # Update cluster centers
        for j in 1:k
            cluster_data = data[assignments .== j]
            if !isempty(cluster_data)
                a = mean(cluster_data)
                means[j] = a
            else
                # Reinitialize empty clusters
                means[j] = data[rand(1:length(data))]
            end
        end
    end
    return means, assignments
end

function logsumexp_over_states_at_last_time(alpha::Array{<:Real,3})
    s = log.(sum(exp.(alpha[:, end, :]), dims=2))  # sum over states at t = T
    return vec(s)  # N-vector
end

function lkl_func(;
	model_params::NamedTuple,
	sim_output::SimulationOutput,
)

	alpha = populate_forward_threads(;
		model_params = model_params,
		sim_output = sim_output,
	)
	# log_sample_contribution = sum_alpha_across_states(alpha[:, :, end]) # using the last timepoint
	log_sample_contribution = sum_alpha_across_states(alpha[:, end, :]) # using the last timepoint
	# log_sample_contribution = logsumexp_over_states_at_last_time(alpha)
	
	# neg_log_lkl = -sum(log_sample_contribution .+ log.(lkl_weights)) # weight it in log space
	neg_log_lkl = -sum(log_sample_contribution) # weight it in log space

	return neg_log_lkl

end

function populate_forward_threads(;
	model_params::NamedTuple,
	sim_output::SimulationOutput,
)

	# unpack 
	sim_params, sim_hyper = unpack(sim_output)
	covariate_idx = convert_covariate_2_df_indices(
		sim_params.covariate_mat_headers,
		sim_hyper.covariate_tup
	)

	bern_cond = haskey(sim_output.observations, :bernoulli_observations)
	gauss_cond = haskey(sim_output.observations, :gaussian_observations)

	N = sim_hyper.N
	if N > Threads.nthreads()
		sample_chunks_idx = Iterators.partition(1:N, N ÷ Threads.nthreads())
	else
		sample_chunks_idx = Iterators.partition(1:N, N)
	end

	tasks = map(sample_chunks_idx) do chunk_idx
		Threads.@spawn populate_forward(;
			model_params = model_params,

			bernoulli_observations = bern_cond ? sim_output.observations.bernoulli_observations[chunk_idx, :, :] : nothing,
			gaussian_observations = gauss_cond ? sim_output.observations.gaussian_observations[chunk_idx, :, :] : nothing,

			covariate_mat = sim_params.covariate_mat[chunk_idx, :],
			covariate_idx = covariate_idx
		)
	end

	alpha_chunks = fetch.(tasks)

	# recombine
	alpha = alpha_chunks[1]
	for i in 2:length(alpha_chunks)
		alpha = cat(alpha, alpha_chunks[i], dims = 1)
	end

	return alpha
end

function precompute_transition(
	beta_transition::Array{<:Real},
	rho_param::Vector{<:Real},
	covariate_mat::Array{<:Union{Missing, Float64}},
)

	# precompute the transition matrix of each patient
	N = size(covariate_mat, 1)
	# I = size(beta_transition, 1) + 1
	I = size(beta_transition, 3)
	log_transition_tensor = zeros(eltype(beta_transition), N, I, I)
	for n in 1:N
		# log_transition_tensor[n, :, :] .= get_transition_mat(beta_transition, patient_mat[n, :])
		# log_transition_tensor[n, :, :] .= get_rho_gamma_2_transition_mat(gamma_transition, rho_param, patient_mat[n, :])
		log_transition_tensor[n, :, :] .= get_rho_beta_transition_mat(beta_transition, rho_param, covariate_mat[n, :])
	end
	log_transition_tensor = log.(log_transition_tensor)
	return log_transition_tensor
end

function populate_forward(;
	model_params::NamedTuple,
	bernoulli_observations = nothing,
	gaussian_observations = nothing,
	covariate_mat::Matrix{Float64},
	covariate_idx::NamedTuple
)

	# precompute transition and emission mats (for speed..)
	log_transition_tensor = precompute_transition(
		model_params.beta_transition,
		model_params.rho_trans,
		covariate_mat[:, covariate_idx[:trans]],
	)
	
	using_bern = haskey(model_params.emissions, :beta_bernoulli)
	using_gauss = haskey(model_params.emissions, :beta_gaussian)

	# bernoulli_probs_tensor = precompute_emission(
	# 	model_params.emissions.beta_bernoulli,
	# 	covariate_mat[:, covariate_idx[:em]],
	# )

	# emission currently cant handle covariates
	bernoulli_probs = using_bern ? get_bernoulli_probs(model_params.emissions.beta_bernoulli, covariate_mat[1, covariate_idx[:em]]) : nothing
	# print(model_params.emissions.beta_gaussian)
	true_gaussian_emission = using_gauss ? convert_gauss_2_true(model_params.emissions.beta_gaussian) : nothing


	N = size(covariate_mat, 1)
	I = size(model_params.beta_transition, 3)

	if using_bern && using_gauss
		@assert size(bernoulli_observations, 2) == size(gaussian_observations, 2) "Bernoulli and Gaussian T differ"
		T = size(bernoulli_observations, 2)
	elseif using_bern
		T = size(bernoulli_observations, 2)
	elseif using_gauss
		T = size(gaussian_observations, 2)
	else
		stop("No observations provided")
	end
	
	# pre allocation
	alpha = zeros(eltype(model_params.beta_initial), N, T, I)
	initial_states = zeros(eltype(model_params.beta_initial), I)
	log_symp_contr::eltype(model_params.beta_initial) = 0.0
	log_trans_prob::eltype(model_params.beta_initial) = 0.0

	# initialisation
	for n in 1:N

		# generate prob matrices
		initial_states = get_rho_beta_initial_states(
			model_params.beta_initial, 
			model_params.rho_initial, 
			covariate_mat[n, covariate_idx[:initial]])
		log_state = log.(initial_states)

		for i in 1:I
			log_symp_contr = get_probs_from_observable(;
				state_no = i,
				bernoulli_obs_vec = using_bern ? bernoulli_observations[n, 1, :] : nothing,
				gaussian_obs_vec = using_gauss ? gaussian_observations[n, 1, :] : nothing,
				bernoulli_probs = bernoulli_probs,
				true_gaussian_emission = true_gaussian_emission
			)
			alpha[n, 1, i] = log_symp_contr + log_state[i]
		end
	end

	# from time = 2
	for t in 2:T
		for n in 1:N
			for i in 1:I

				# transition to new states
				log_trans_prob = log_space_transition_state(
					state_no = i,
					log_state_probs = alpha[n, t-1, :],
					log_transition_mat = log_transition_tensor[n, :, :],
					n_states = I,
				)

				# observation
				log_symp_contr = get_probs_from_observable(;
					state_no = i,
					bernoulli_obs_vec = using_bern ? bernoulli_observations[n, t, :] : nothing,
					gaussian_obs_vec = using_gauss ? gaussian_observations[n, t, :] : nothing,
					bernoulli_probs = bernoulli_probs,
					true_gaussian_emission = true_gaussian_emission
				)

				alpha[n, t, i] = log_trans_prob + log_symp_contr
			end
		end
	end

	return alpha
end

function get_probs_from_observable(;
	state_no::Int64,
	bernoulli_obs_vec::Union{Vector{Int64}, Nothing},
	gaussian_obs_vec::Union{Vector{Float64}, Nothing},
	bernoulli_probs::Union{Matrix{<:Real}, Nothing},
	true_gaussian_emission::Union{NamedTuple, Nothing}
)
	# get unormalised vector fof proabbility latent states, using observables but NOT state probs
	log_symp_contr = 0

	if !isnothing(bernoulli_obs_vec)
		# bernoulli
		log_symp_contr += get_bernoulli_accumalator(;
			state_no = state_no,
			bernoulli_obs_vec = bernoulli_obs_vec,
			bernoulli_probs = bernoulli_probs
		)
	end

	if !isnothing(gaussian_obs_vec)
		# gaussian
		log_symp_contr += get_gaussian_accumalator(;
			state_no = state_no,
			gaussian_obs_vec = gaussian_obs_vec,
			true_gaussian_emission = true_gaussian_emission
		)
	end


	return log_symp_contr
end

function get_bernoulli_accumalator(;
	state_no::Int64,
	bernoulli_obs_vec::Vector{Int64},
	bernoulli_probs::Matrix{<:Real}
)

	n_sympt = length(bernoulli_obs_vec)
	probs_from_states = zeros(eltype(bernoulli_probs), n_sympt)
	for symptom_idx in 1:length(bernoulli_obs_vec)

		probs_from_states[symptom_idx] = get_bernoulli_val(
			from_state = state_no,
			to_observable = symptom_idx,
			observable_instance = bernoulli_obs_vec[symptom_idx],
			bernoulli_probs = bernoulli_probs
		)
	end

	log_symp_contr = sum(log.(probs_from_states))

	return log_symp_contr
end

function get_bernoulli_val(;
	from_state::Int64,
	to_observable::Int64,
	observable_instance::Int64,
	bernoulli_probs::Matrix{<:Real},
)
	output::eltype(bernoulli_probs) = 0.0
	if observable_instance == 1
		output = bernoulli_probs[from_state, to_observable]
	elseif observable_instance == 0
		output = 1 - bernoulli_probs[from_state, to_observable]
	elseif observable_instance == -1
		output = 1 # basically skip the observation
	end

	return output
end

function get_gaussian_accumalator(;
	state_no::Int64,
	gaussian_obs_vec::Vector{Float64},
	true_gaussian_emission::NamedTuple
)

	# accumulate the contributions from each symptom, then normalise
	n_obs_symp = length(gaussian_obs_vec)
	probs_from_states = zeros(eltype(true_gaussian_emission.means), n_obs_symp)
	for symptom_idx in 1:n_obs_symp

		probs_from_states[symptom_idx] = get_gauss_val(
			from_state = state_no,
			to_observable = symptom_idx,
			observable_instance = gaussian_obs_vec[symptom_idx],
			true_gaussian_emission = true_gaussian_emission)
	end

	log_symp_contr = sum(log.(probs_from_states))

	return log_symp_contr
end

function get_gauss_val(;
	from_state::Int64,
	to_observable::Int64,
	observable_instance::Float64,
	true_gaussian_emission::NamedTuple,
)
	mu = true_gaussian_emission.means[from_state, to_observable]
	standard_dev = true_gaussian_emission.stds[from_state, to_observable]

	val = get_pdf_norm(
		observable_instance; 
		mu = mu, standard_dev = standard_dev
	)

	return val
end

function get_pdf_norm(
	x; mu, standard_dev
)
	
	if x != -1.0
		val = (1 / (sqrt(2 * pi * standard_dev^2))) * exp(-0.5 * ((x - mu) / standard_dev)^2 ) # pdf normal distribution
	else
		val = 1 # basically skip the observation
	end

	# handle low variance
	val = maximum([val, 1e-6])

	return val
end

function log_space_transition_state(;
	state_no::Int64,
	log_state_probs::Vector{<:Real},
	log_transition_mat::Matrix{<:Real},
	n_states = n_states,
)
	# move one step in time, but in log space
	# exp_next_prob = 0
	# for from_state_j in 1:n_states
	# 	exp_next_prob = exp_next_prob + exp(log_transition_mat[from_state_j, state_no] + log_state_probs[from_state_j])
	# end

	# log_next_probs = log(exp_next_prob)

	# return log_next_probs

	v = log_transition_mat[:, state_no] .+ log_state_probs
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end

function sum_alpha_across_states(
	alpha_T::Matrix{<:Real},
)
	out = log.(sum(exp.(alpha_T), dims = 2))

	return vec(out)
end

# uncertainity
# using NLSolversBase, Measurements, LinearAlgebra
using NLSolversBase

function get_hessian(
    funct,
    est_params::Vector{<:Real},
)
    diff_obj = TwiceDifferentiable(
        funct,
        est_params, 
        autodiff=:forward)
    numerical_hessian = hessian!(diff_obj, est_params)

    return numerical_hessian
end

function get_se(
    funct,
    est_params::Vector{<:Real},
    lambda::Float64
)

    if lambda == 0.0
        # if there's no regularisation, get the covariance mat
        hessian = get_hessian(
            funct,
            est_params,
        )
        try
            var_cov_matrix = inv(hessian)
            std_params = sqrt.(diag(var_cov_matrix))
        catch
            print("Needed adjustment")

            # now make adjustment
            eps_adj = 2 * abs(minimum(real(eigen(hessian).values)))
            adjusted_hesian = hessian + I(size(hessian, 1)) .* eps_adj
            var_cov_matrix = inv(adjusted_hesian)
            std_params = sqrt.(diag(var_cov_matrix))
        end

    else
        std_params = zeros(length(est_params))
    end

    return std_params
end

function get_se(
    hessian::Matrix{Float64}
)
    # std_params = zeros(size(hessian, 1))
    std_params = Vector{Float64}(undef, size(hessian, 1))
    try
        var_cov_matrix = inv(hessian)
        std_params = sqrt.(diag(var_cov_matrix))
    catch

        try
            # now make adjustment
            eps_adj = 2 * abs(minimum(real(eigen(hessian).values)))
            adjusted_hesian = hessian + I(size(hessian, 1)) .* eps_adj
            var_cov_matrix = inv(adjusted_hesian)
            
            std_params = sqrt.(diag(var_cov_matrix))
        catch
            std_params = Vector{Float64}(undef, size(hessian, 1))
        end
    end

    return std_params
end